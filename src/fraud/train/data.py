"""Loads feature rows for a day range straight from Parquet via DuckDB."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from fraud.config import project_path
from fraud.features import feature_names


def _con(cfg: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg['features']['duckdb_memory_limit']}'")
    con.execute(f"SET threads={cfg['features']['duckdb_threads']}")
    return con


def _glob(cfg: dict, bucket: int | None = None) -> str:
    base = project_path(cfg["features"]["dir"])
    return (base / (f"bucket={bucket:02d}" if bucket is not None else "*") / "*.parquet").as_posix()


def days_range(cfg: dict, key: str) -> tuple[int, int]:
    lo, hi = cfg["train"][key]
    return lo, hi


def count_legit(cfg: dict, days: tuple[int, int]) -> int:
    return _con(cfg).execute(
        f"SELECT count(*) FROM '{_glob(cfg)}' WHERE is_fraud = 0 AND day BETWEEN {days[0]} AND {days[1]}"
    ).fetchone()[0]


def load_sampled(cfg: dict, days: tuple[int, int], legit_rows: int) -> pd.DataFrame:
    """All fraud rows in the day range + a deterministic hash sample of ~legit_rows legit rows."""
    keep = min(legit_rows / count_legit(cfg, days), 1.0)
    cols = ", ".join(feature_names())
    return _con(cfg).execute(f"""
        SELECT transaction_id, day, kind, is_fraud, {cols}
        FROM '{_glob(cfg)}'
        WHERE day BETWEEN {days[0]} AND {days[1]}
          AND (is_fraud = 1 OR hash(transaction_id) % 1000000 < {int(keep * 1_000_000)})
    """).df()


def load_bucket(cfg: dict, bucket: int, days: tuple[int, int]) -> pd.DataFrame:
    """Every row (no sampling) of one card bucket in the day range; used to score the test set in chunks."""
    cols = ", ".join(feature_names())
    return _con(cfg).execute(f"""
        SELECT transaction_id, day, kind, is_fraud, {cols}
        FROM '{_glob(cfg, bucket)}' WHERE day BETWEEN {days[0]} AND {days[1]}
    """).df()


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[feature_names()].to_numpy(np.float32), df["is_fraud"].to_numpy(np.int8)
