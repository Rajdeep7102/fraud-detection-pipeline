"""Loads config.yaml and resolves repo-relative paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    """Where the trained model lives: $FRAUD_MODEL_DIR, else artifacts/ (a model you trained yourself), else the
    ready-made models/demo/ that ships with the repo, so a fresh clone can score transactions with no training."""
    env = os.environ.get("FRAUD_MODEL_DIR")
    if env:
        return Path(env)
    for candidate in (ROOT / "artifacts", ROOT / "models" / "demo"):
        if (candidate / "model.joblib").exists():
            return candidate
    return ROOT / "models" / "demo"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
