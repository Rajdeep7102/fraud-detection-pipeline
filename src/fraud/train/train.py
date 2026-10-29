"""Trains the Random Forest and freezes everything serving needs into artifacts/.

Split (time-based, by day from the start date):
  train  days 0-62   all fraud + a hash sample of legit rows (a full 30M-row forest will not fit in 6 GB)
  valid  days 63-69  natural class balance; used ONLY to choose the decision threshold
  test   days 70-89  never touched here; scored once by evaluate.py

    python -m fraud.train.train [--trees 200] [--legit-rows 1500000] [--depth 16] [--min-leaf 5]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraud.config import load_config, project_path
from fraud.features import feature_names
from fraud.monitoring import drift
from fraud.train import data


def pick_threshold(y: np.ndarray, p: np.ndarray, target_precision: float) -> dict:
    """Highest-recall threshold whose validation precision >= target; else the best-F1 threshold."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = prec >= target_precision
    if ok.any():
        i = int(np.flatnonzero(ok)[np.argmax(rec[ok])])
        met = True
    else:
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        i, met = int(np.argmax(f1)), False
    return {"threshold": float(thr[i]), "valid_precision": float(prec[i]), "valid_recall": float(rec[i]),
            "target_precision_met": met}


def make_baseline(valid_x: np.ndarray, valid_p: np.ndarray) -> dict:
    """Bins + expected shares per feature and for the score (natural distribution), for PSI drift checks."""
    return {"features": {n: drift.describe(valid_x[:, j]) for j, n in enumerate(feature_names())},
            "score": drift.describe(valid_p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int)
    ap.add_argument("--legit-rows", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--min-leaf", type=int)
    ap.add_argument("--out", default=None, help="artifact dir (default: train.model_dir)")
    ap.add_argument("--resume", action="store_true", help="reuse model.joblib in --out instead of refitting")
    args = ap.parse_args()

    cfg = load_config()
    tc = cfg["train"]
    rf = dict(tc["rf"])
    rf["n_estimators"] = args.trees or rf["n_estimators"]
    rf["max_depth"] = args.depth or rf["max_depth"]
    rf["min_samples_leaf"] = args.min_leaf or rf["min_samples_leaf"]
    legit_rows = args.legit_rows or tc["legit_sample_rows"]
    out = project_path(args.out or tc["model_dir"])

    t0 = time.perf_counter()
    train = data.load_sampled(cfg, data.days_range(cfg, "train_days"), legit_rows)
    valid = data.load_sampled(cfg, data.days_range(cfg, "valid_days"), 10**12)   # all rows, natural balance
    x_tr, y_tr = data.xy(train)
    x_va, y_va = data.xy(valid)
    print(f"train {len(train):,} rows ({int(y_tr.sum()):,} fraud) | valid {len(valid):,} rows "
          f"({int(y_va.sum()):,} fraud) | loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    saved = out / "model.joblib"
    if args.resume and saved.exists():
        # Recovery path: reuse a forest saved by an interrupted run (same seed + params => same model).
        model, fit_s = joblib.load(saved), None
        assert model.n_estimators == rf["n_estimators"], "saved forest does not match config; refit instead"
        print(f"reusing saved forest ({model.n_estimators} trees)", flush=True)
    else:
        t1 = time.perf_counter()
        model = RandomForestClassifier(random_state=cfg["seed"], **rf).fit(x_tr, y_tr)
        fit_s = time.perf_counter() - t1
        print(f"fit {rf['n_estimators']} trees in {fit_s:.0f}s", flush=True)

    p_va = model.predict_proba(x_va)[:, 1]
    chosen = pick_threshold(y_va, p_va, tc["target_precision"])
    print("validation:", json.dumps(chosen), f"PR-AUC={average_precision_score(y_va, p_va):.4f}", flush=True)

    if not args.resume:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        tmp = saved.with_suffix(".tmp")
        joblib.dump(model, tmp, compress=3)
        tmp.replace(saved)   # atomic: an interrupted run never leaves a half-written model.joblib
    size_mb = (out / "model.joblib").stat().st_size / 1e6
    meta = {
        "model_version": f"rf-{time.strftime('%Y%m%d%H%M')}",
        "features": feature_names(),
        "rf_params": rf,
        "train_rows": int(len(train)), "train_fraud_rows": int(y_tr.sum()),
        "valid_rows": int(len(valid)), "valid_fraud_rows": int(y_va.sum()),
        "valid_pr_auc": float(average_precision_score(y_va, p_va)),
        "model_size_mb": round(size_mb, 1), "fit_seconds": None if fit_s is None else round(fit_s, 1),
        **chosen,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "baseline.json").write_text(json.dumps(make_baseline(x_va, p_va), indent=2))
    spec = project_path("src/fraud/features/feature_spec.yaml")
    shutil.copy(spec, out / "feature_spec.yaml")
    print(f"saved model ({size_mb:.0f} MB) + meta + baseline to {out}")
    print(yaml.safe_dump({k: meta[k] for k in ("model_version", "threshold", "valid_precision", "valid_recall")}))


if __name__ == "__main__":
    main()
