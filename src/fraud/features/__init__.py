"""Feature extraction: DuckDB window SQL plus the shared feature spec."""
from __future__ import annotations

from importlib import resources

import yaml


def feature_names() -> list[str]:
    spec = yaml.safe_load(resources.files("fraud.features").joinpath("feature_spec.yaml").read_text("utf-8"))
    return [f["name"] for f in spec["features"]]


def feature_sql() -> str:
    return resources.files("fraud.features").joinpath("sql/features.sql").read_text("utf-8")
