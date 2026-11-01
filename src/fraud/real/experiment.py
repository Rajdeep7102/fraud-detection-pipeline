"""Real-data validation on IEEE-CIS: does the pipeline's approach hold up outside synthetic data?

Time-based split on TransactionDT (train first 60% of rows, validation next 15%, test last 25%). Thresholds are
chosen on validation only. Three feature sets are compared:
  A  the 12 clues (real counterparts of the synthetic pipeline's features)
  B  the raw columns the dataset provides (amount, product, card/address fields, C1-14, D1-15, M1-9)
  C  clues + raw columns
Then four ways of handling class imbalance (none, class weights, random undersampling, SMOTE) are compared on A.
SMOTE and undersampling run inside an imblearn Pipeline, so they touch the training rows only.

    python -m fraud.real.experiment
"""
from __future__ import annotations

import json
import time

import duckdb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fraud.real.features import CLUES, OUT as CLUES_PATH, SRC
from fraud.config import ROOT
from fraud.train.train import pick_threshold

SEED = 42
REPORT = ROOT / "reports" / "real"
RAW = (["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]
       + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)])
RAW_CAT = ["ProductCD", "card4", "card6"] + [f"M{i}" for i in range(1, 10)]
PRECISION_TARGET = 0.80       # real fraud is much harder than the synthetic data; 0.94 is not reachable here
BUDGET = 0.01                 # "we can review 1% of transactions"


def load() -> pd.DataFrame:
    cols = ", ".join(f"t.{c}" for c in RAW + RAW_CAT)
    clues = ", ".join(f"c.{c}" for c in CLUES)
    df = duckdb.sql(f"""
        SELECT c.TransactionID, c.isFraud, c.ts, {clues}, {cols}
        FROM '{CLUES_PATH.as_posix()}' c JOIN '{SRC.as_posix()}' t USING (TransactionID)
        ORDER BY c.ts""").df()
    for c in RAW_CAT:   # category codes; missing stays missing (the forest handles NaN natively)
        codes = df[c].astype("category").cat.codes.astype("float64")
        df[c] = codes.where(codes >= 0, np.nan)
    return df


def split(df: pd.DataFrame):
    n = len(df)
    i1, i2 = int(n * 0.60), int(n * 0.75)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def rf(class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=5, max_features="sqrt",
                                  class_weight=class_weight, n_jobs=-1, random_state=SEED)


def make(strategy: str):
    if strategy == "none":
        return rf()
    if strategy == "class_weight":
        return rf("balanced_subsample")
    if strategy == "undersample":
        return Pipeline([("under", RandomUnderSampler(sampling_strategy=0.1, random_state=SEED)), ("rf", rf())])
    if strategy == "smote":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("smote", SMOTE(sampling_strategy=0.5, random_state=SEED)), ("rf", rf())])
    raise ValueError(strategy)


def bootstrap_ci(y, p, fn, n=100):
    rng = np.random.default_rng(SEED)
    vals = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(np.quantile(vals, 0.025)), 4), round(float(np.quantile(vals, 0.975)), 4)]


def evaluate(model, cols, tr, va, te) -> dict:
    t0 = time.perf_counter()
    model.fit(tr[cols], tr["isFraud"])
    pv, pt = model.predict_proba(va[cols])[:, 1], model.predict_proba(te[cols])[:, 1]
    yv, yt = va["isFraud"].to_numpy(), te["isFraud"].to_numpy()

    op = pick_threshold(yv, pv, PRECISION_TARGET)
    flag = pt >= op["threshold"]
    tp = int((flag & (yt == 1)).sum())
    budget_thr = float(np.quantile(pv, 1 - BUDGET))
    bflag = pt >= budget_thr
    btp = int((bflag & (yt == 1)).sum())
    return {
        "test_roc_auc": round(float(roc_auc_score(yt, pt)), 4), "test_roc_auc_95ci": bootstrap_ci(yt, pt, roc_auc_score),
        "test_pr_auc": round(float(average_precision_score(yt, pt)), 4), "test_pr_auc_95ci": bootstrap_ci(yt, pt, average_precision_score),
        "precision_target": PRECISION_TARGET, "precision_target_met_on_validation": op["target_precision_met"],
        "at_validation_threshold": {"flagged_pct": round(100 * float(flag.mean()), 2),
                                    "precision": round(tp / max(int(flag.sum()), 1), 4),
                                    "recall": round(tp / int(yt.sum()), 4)},
        "at_1pct_review_budget": {"flagged_pct": round(100 * float(bflag.mean()), 2),
                                  "precision": round(btp / max(int(bflag.sum()), 1), 4),
                                  "recall": round(btp / int(yt.sum()), 4)},
        "fit_seconds": round(time.perf_counter() - t0), "_pt": pt, "_yt": yt,
        "_model": model,
    }


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    df = load()
    tr, va, te = split(df)
    day = lambda d: (round(d["ts"].min() / 86400, 1), round(d["ts"].max() / 86400, 1))  # noqa: E731
    info = {"rows": {"train": len(tr), "valid": len(va), "test": len(te)},
            "fraud_rate_pct": {k: round(100 * d["isFraud"].mean(), 2) for k, d in (("train", tr), ("valid", va), ("test", te))},
            "days": {"train": day(tr), "valid": day(va), "test": day(te)}}
    print(json.dumps(info), flush=True)

    sets = {"A_clues": CLUES, "B_raw_columns": RAW + RAW_CAT, "C_clues_plus_raw": CLUES + RAW + RAW_CAT}
    feature_sets, curves, importances = {}, {}, {}
    for name, cols in sets.items():
        r = evaluate(make("class_weight"), cols, tr, va, te)
        curves[name] = precision_recall_curve(r["_yt"], r["_pt"])[:2]
        importances[name] = dict(sorted(zip(cols, r["_model"].feature_importances_.round(4).tolist()), key=lambda kv: -kv[1])[:8])
        feature_sets[name] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(name, json.dumps(feature_sets[name]), flush=True)

    imbalance = {}
    for strat in ("none", "class_weight", "undersample", "smote"):
        r = evaluate(make(strat), CLUES, tr, va, te)
        imbalance[strat] = {k: v for k, v in r.items() if not k.startswith("_")}
        print("A_clues /", strat, json.dumps(imbalance[strat]), flush=True)

    out = {"dataset": "IEEE-CIS Fraud Detection (Vesta), labelled training file, 590,540 real transactions",
           "split": info, "model": "RandomForest(200 trees, depth 16), thresholds chosen on validation only",
           "feature_sets": feature_sets, "imbalance_on_A_clues": imbalance, "top_features": importances,
           "test_base_rate_pr_auc": round(float(te["isFraud"].mean()), 4)}
    (REPORT / "metrics.json").write_text(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, (pr, rc) in curves.items():
        ax.plot(rc, pr, label=f"{name} (AP {feature_sets[name]['test_pr_auc']:.2f})")
    ax.axhline(out["test_base_rate_pr_auc"], color="grey", ls=":", label="no skill")
    ax.set(xlabel="Recall", ylabel="Precision", title="Real data (IEEE-CIS), test period", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT / "pr_curve_real.png", dpi=140)
    print("done")


if __name__ == "__main__":
    main()
