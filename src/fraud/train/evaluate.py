"""Scores the untouched test days once, chunked by card bucket, with the frozen threshold.

Writes reports/metrics.json plus PR-curve, confusion-matrix and feature-importance plots.

    python -m fraud.train.evaluate [--model-dir artifacts]
"""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score)  # noqa: E402

from fraud.config import load_config, project_path  # noqa: E402
from fraud.features import feature_names  # noqa: E402
from fraud.train import data  # noqa: E402


def bootstrap_ci(y: np.ndarray, flag: np.ndarray, n: int = 200, seed: int = 0) -> dict:
    """Resample whole fraud/non-fraud rows to get a 95% interval on precision and recall."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    prec, rec = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        tp = int((flag[idx] & (y[idx] == 1)).sum())
        fp = int((flag[idx] & (y[idx] == 0)).sum())
        prec.append(tp / max(tp + fp, 1))
        rec.append(tp / len(pos))
    q = lambda a: [round(float(np.quantile(a, 0.025)), 4), round(float(np.quantile(a, 0.975)), 4)]  # noqa: E731
    return {"precision_95ci": q(prec), "recall_95ci": q(rec)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()
    cfg = load_config()
    mdir = project_path(args.model_dir or cfg["train"]["model_dir"])
    rdir = project_path(cfg["train"]["reports_dir"])
    rdir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(mdir / "model.joblib")
    meta = json.loads((mdir / "model_meta.json").read_text())
    thr = meta["threshold"]
    days = data.days_range(cfg, "test_days")

    ys, ps, kinds, interesting = [], [], [], []
    for b in range(cfg["data"]["n_buckets"]):
        df = data.load_bucket(cfg, b, days)
        x, y = data.xy(df)
        p_b = model.predict_proba(x)[:, 1].astype(np.float32)
        ps.append(p_b)
        ys.append(y)
        kinds.append(df["kind"].to_numpy())
        # Keep every flagged or truly fraudulent row (about 0.5% of rows) so reports can attach amounts later.
        keep = (p_b >= thr) | (y == 1)
        interesting.append(pd.DataFrame({"transaction_id": df["transaction_id"].to_numpy()[keep],
                                         "kind": df["kind"].to_numpy()[keep], "is_fraud": y[keep],
                                         "fraud_score": p_b[keep], "flagged": p_b[keep] >= thr}))
        print(f"scored bucket {b:02d} ({sum(map(len, ys)):,} rows)", flush=True)
    pd.concat(interesting).to_parquet(rdir / "test_flagged_or_fraud.parquet", index=False)
    y, p, kind = np.concatenate(ys), np.concatenate(ps), np.concatenate(kinds)
    flag = p >= thr

    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    by_kind = {k: {"rows": int((kind == k).sum()), "flagged": int(flag[kind == k].sum()),
                   "flag_rate": round(float(flag[kind == k].mean()), 4)}
               for k in ("card_testing", "account_takeover", "stealth", "travel", "normal")}
    for k in ("card_testing", "account_takeover", "stealth"):
        by_kind[k]["recall"] = by_kind[k].pop("flag_rate")

    metrics = {
        "split": {"test_days": list(days), "rows": int(len(y)), "fraud_rows": int(y.sum()),
                  "fraud_rate": round(float(y.mean()), 5)},
        "threshold": thr, "threshold_chosen_on": "validation days (never the test set)",
        "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "by_kind": by_kind,
        **bootstrap_ci(y, flag),
        "model_version": meta["model_version"], "model_size_mb": meta["model_size_mb"],
        "validation": {k: meta[k] for k in ("valid_precision", "valid_recall", "valid_pr_auc")},
    }
    (rdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec)
    ax.scatter([metrics["recall"]], [metrics["precision"]], color="C3", zorder=3, label=f"threshold {thr:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Test PR curve (AP {metrics['pr_auc']:.3f})",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(rdir / "pr_curve.png", dpi=140)

    imp = pd.Series(model.feature_importances_, index=feature_names()).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax)
    ax.set(title="Feature importance (mean decrease in impurity)")
    fig.tight_layout()
    fig.savefig(rdir / "feature_importance.png", dpi=140)
    imp.sort_values(ascending=False).round(4).to_json(rdir / "feature_importance.json", indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
