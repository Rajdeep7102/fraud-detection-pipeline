"""Converts the IEEE-CIS CSV files to typed Parquet once, so later steps read in seconds, not minutes.

Input (extracted by hand from the Kaggle zip, which is not redistributed here):
    data/real/raw/train_transaction.csv   data/real/raw/train_identity.csv
Output:
    data/real/parquet/transactions.parquet   (transaction + identity columns, joined on TransactionID)

    python -m fraud.real.prepare
"""
from __future__ import annotations

import time

import duckdb

from fraud.config import ROOT

RAW = ROOT / "data" / "real" / "raw"
OUT = ROOT / "data" / "real" / "parquet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory='{(ROOT / 'data' / 'duckdb_tmp').as_posix()}'")
    t0 = time.perf_counter()
    con.execute(f"""
        COPY (
            SELECT t.*, i.DeviceType, i.DeviceInfo, i.id_30 AS os, i.id_31 AS browser, i.id_33 AS screen_res
            FROM read_csv('{(RAW / 'train_transaction.csv').as_posix()}', sample_size=-1) t
            LEFT JOIN read_csv('{(RAW / 'train_identity.csv').as_posix()}', sample_size=-1) i USING (TransactionID)
            ORDER BY t.TransactionDT
        ) TO '{(OUT / 'transactions.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    n, fraud = con.execute(f"SELECT count(*), sum(isFraud) FROM '{(OUT / 'transactions.parquet').as_posix()}'").fetchone()
    print(f"wrote {n:,} rows ({int(fraud):,} fraud) in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
