"""Runs features.sql over every card bucket and writes data/features/bucket=NN/part-0.parquet.

Each bucket holds whole cards, so windows never cross buckets and memory stays bounded.

    python -m fraud.features.build [--buckets 0,1,2]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import duckdb

from fraud.config import load_config, project_path
from fraud.features import feature_names, feature_sql


def build(cfg: dict, buckets: list[int] | None = None) -> dict:
    raw = project_path(cfg["data"]["raw_dir"])
    out = project_path(cfg["features"]["dir"])
    n_buckets = cfg["data"]["n_buckets"]
    buckets = list(range(n_buckets)) if buckets is None else buckets
    if buckets == list(range(n_buckets)) and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg['features']['duckdb_memory_limit']}'")
    con.execute(f"SET threads={cfg['features']['duckdb_threads']}")
    con.execute(f"SET temp_directory='{(project_path('data') / 'duckdb_tmp').as_posix()}'")
    cards = (raw / "cards.parquet").as_posix()
    con.execute(f"CREATE VIEW cards AS SELECT * FROM read_parquet('{cards}')")
    sql = feature_sql()

    rows = fraud = 0
    t0 = time.perf_counter()
    for b in buckets:
        src = (raw / "transactions" / f"bucket={b:02d}" / "part-0.parquet").as_posix()
        con.execute(f"CREATE OR REPLACE VIEW txn AS SELECT * FROM read_parquet('{src}')")
        dest = out / f"bucket={b:02d}"
        dest.mkdir(exist_ok=True)
        target = (dest / "part-0.parquet").as_posix()
        con.execute(f"COPY ({sql}) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n, f = con.execute(f"SELECT count(*), sum(is_fraud) FROM read_parquet('{target}')").fetchone()
        rows, fraud = rows + n, fraud + int(f)
        print(f"bucket {b:02d}: {n:,} rows  ({time.perf_counter() - t0:,.0f}s)", flush=True)

    summary = {"rows": rows, "fraud_rows": fraud, "features": feature_names(),
               "seconds": round(time.perf_counter() - t0, 1)}
    if len(buckets) == n_buckets:
        (out / "_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default=None, help="comma-separated bucket ids (default: all)")
    args = ap.parse_args()
    buckets = [int(x) for x in args.buckets.split(",")] if args.buckets else None
    print(json.dumps(build(load_config(), buckets), indent=2))


if __name__ == "__main__":
    main()
