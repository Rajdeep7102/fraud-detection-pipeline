"""Sends held-out test-day transactions to the live endpoint, as normal or deliberately drifted traffic.

    python monitoring/replay_traffic.py --rows 5000                 # natural traffic
    python monitoring/replay_traffic.py --rows 5000 --drift         # shifted velocity / amount / geography

Each request carries CustomAttributes=replay-<label>-<n>, which the container uses as the audit log
request_id, so a replay's lines can be found (and told apart from other traffic) later.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import duckdb
import numpy as np
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
BATCH = 50
RETRYABLE = ("ModelError", "ThrottlingException", "ServiceUnavailable", "ModelNotReadyException",
             "InternalFailure", "ReadTimeoutError")


def load_rows(cfg: dict, n: int, seed: int, test_days: tuple[int, int]) -> "duckdb.DuckDBPyRelation":
    import pandas as pd  # noqa: F401  (duckdb .df())
    feats = [f["name"] for f in yaml.safe_load((ROOT / "src/fraud/features/feature_spec.yaml").read_text())["features"]]
    glob = (ROOT / cfg["features"]["dir"] / "*" / "*.parquet").as_posix()
    # The WHERE must sit in a subquery: DuckDB applies USING SAMPLE before a same-level WHERE, which would
    # sample from all 90 days and then keep only the test days (about a fifth of the requested rows).
    return duckdb.sql(f"""
        SELECT * FROM (SELECT transaction_id, is_fraud, {', '.join(feats)} FROM '{glob}'
                       WHERE day BETWEEN {test_days[0]} AND {test_days[1]})
        USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})""").df(), feats


def apply_drift(df, rng: np.random.Generator):
    """A plausible shift: a burst of activity, larger amounts, more foreign and new-merchant traffic."""
    d = df.copy()
    for c in ("txn_count_1h", "txn_count_24h", "distinct_merchants_24h"):
        d[c] = np.ceil(d[c] * rng.uniform(2.5, 4.0, len(d))).astype(int)
    d["amount_sum_24h"] = d["amount_sum_24h"] * rng.uniform(2.0, 3.5, len(d))
    d["amount_to_avg_30d"] = d["amount_to_avg_30d"] * rng.uniform(1.5, 3.0, len(d))
    d["secs_since_last_txn"] = d["secs_since_last_txn"] * rng.uniform(0.05, 0.4, len(d))
    d["is_foreign_country"] = np.where(rng.random(len(d)) < 0.30, 1, d["is_foreign_country"])
    d["is_new_merchant"] = np.where(rng.random(len(d)) < 0.50, 1, d["is_new_merchant"])
    return d


def invoke(rt, endpoint: str, instances: list[dict], request_id: str) -> tuple[list[dict], float, int]:
    body = json.dumps({"instances": instances})
    for attempt in range(6):
        t = time.perf_counter()
        try:
            r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Body=body,
                                   CustomAttributes=request_id)
            return json.loads(r["Body"].read())["predictions"], time.perf_counter() - t, attempt
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in RETRYABLE or attempt == 5:
                raise
        except Exception as e:  # noqa: BLE001 - botocore read timeouts surface as non-ClientError
            if type(e).__name__ not in RETRYABLE or attempt == 5:
                raise
        time.sleep(min(2 ** attempt, 15) + random.random())
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--label", default=None, help="request_id label (default: drift|normal)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    lo, hi = cfg["train"]["test_days"]
    df, feats = load_rows(cfg, args.rows, args.seed, (lo, hi))
    if args.drift:
        df = apply_drift(df, np.random.default_rng(args.seed))
    label = args.label or ("drift" if args.drift else "normal")
    run = time.strftime("%H%M%S")

    rt = boto3.client("sagemaker-runtime", region_name=cfg["aws"]["region"],
                      config=Config(read_timeout=70, retries={"max_attempts": 0}))
    batches = []
    for i in range(0, len(df), BATCH):
        chunk = df.iloc[i:i + BATCH]
        inst = [{"transaction_id": int(r["transaction_id"]), **{f: float(r[f]) for f in feats}}
                for _, r in chunk.iterrows()]
        batches.append((inst, chunk["is_fraud"].to_numpy(), f"replay-{label}-{run}-{i // BATCH:04d}"))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda b: invoke(rt, cfg["aws"]["endpoint_name"], b[0], b[2]), batches))
    secs = time.perf_counter() - t0

    flags = np.concatenate([[p["decision"] == "FRAUD" for p in preds] for preds, _, _ in results])
    labels = np.concatenate([b[1] for b in batches]).astype(bool)
    lat = np.array([r[1] for r in results])
    tp, fp = int((flags & labels).sum()), int((flags & ~labels).sum())
    print(json.dumps({
        "label": label, "run": run, "rows": len(df), "requests": len(batches), "seconds": round(secs, 1),
        "retried_requests": int(sum(1 for r in results if r[2] > 0)),
        "request_latency_s": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2)},
        "flag_rate": round(float(flags.mean()), 5), "flagged": int(flags.sum()),
        "true_fraud_rows": int(labels.sum()), "flagged_true_fraud": tp, "flagged_legit": fp,
        "note": "labels shown only for replay sanity; drifted rows are synthetic, so precision is not meaningful there",
    }, indent=2))


if __name__ == "__main__":
    main()
