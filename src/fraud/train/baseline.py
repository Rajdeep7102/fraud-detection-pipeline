"""Rebuilds artifacts/baseline.json from the saved model and the validation days, leaving the model,
threshold and model_meta.json untouched (so the deployed model_version stays valid).

    python -m fraud.train.baseline
"""
from __future__ import annotations

import json

import joblib

from fraud.config import load_config, project_path
from fraud.train import data
from fraud.train.train import make_baseline


def main() -> None:
    cfg = load_config()
    mdir = project_path(cfg["train"]["model_dir"])
    model = joblib.load(mdir / "model.joblib")
    valid = data.load_sampled(cfg, data.days_range(cfg, "valid_days"), 10**12)   # all rows, natural balance
    x, _ = data.xy(valid)
    baseline = make_baseline(x, model.predict_proba(x)[:, 1])
    (mdir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    for name, d in {**baseline["features"], "score": baseline["score"]}.items():
        print(f"{name:24s} {d['type']:12s} bins={len(d['expected'])}")


if __name__ == "__main__":
    main()
