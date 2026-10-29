"""Computes PSI for every feature and the fraud_score from recent audit-log lines, and publishes the results.

Reads the last N minutes of JSON audit lines via CloudWatch Logs Insights (deduplicated on
request_id + transaction_id, because log delivery is not exactly-once), compares each feature's distribution
with artifacts/baseline.json, and publishes to namespace FraudDetection:
  PSI{Feature=<name>} for each feature and fraud_score, MaxFeaturePSI, DriftSampleSize.

    python monitoring/drift_job.py --minutes 60
    python monitoring/drift_job.py --minutes 30 --request-id-prefix replay-drift --dry-run

Insights returns at most 10,000 rows per query; beyond that the newest rows are used and a warning is printed.
Run it on a schedule (EventBridge -> Lambda) in production; here it is run by hand.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import boto3
import numpy as np
import yaml

from fraud.monitoring import drift

ROOT = Path(__file__).resolve().parents[1]
NS = "FraudDetection"
INSIGHTS_LIMIT = 10_000
STATUS = ((0.2, "SIGNIFICANT"), (0.1, "moderate"), (0.0, "stable"))


def log(msg: str) -> None:
    print(f"[drift] {msg}", flush=True)


def fetch(logs, group: str, features: list[str], minutes: int, prefix: str | None) -> list[dict]:
    cols = ", ".join(f"features.{f} as {f}" for f in features)
    where = 'event = "prediction"' + (f' and request_id like /^{prefix}/' if prefix else "")
    # dedup must be the LAST command in an Insights query; the row cap is the API's `limit` argument below.
    query = (f"fields @timestamp, request_id, transaction_id, fraud_score, {cols} "
             f"| filter {where} | sort @timestamp desc | dedup request_id, transaction_id")
    end = int(time.time())
    qid = logs.start_query(logGroupName=group, startTime=end - minutes * 60, endTime=end,
                           queryString=query, limit=INSIGHTS_LIMIT)["queryId"]
    while True:
        r = logs.get_query_results(queryId=qid)
        if r["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
        time.sleep(2)
    if r["status"] != "Complete":
        raise SystemExit(f"Insights query {r['status']}")
    return [{c["field"]: c["value"] for c in row} for row in r["results"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--request-id-prefix", default=None, help="only rows whose request_id starts with this")
    ap.add_argument("--min-samples", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="compute and print, publish nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    region = cfg["aws"]["region"]
    baseline = json.loads((ROOT / "artifacts" / "baseline.json").read_text())
    features = list(baseline["features"])
    session = boto3.Session(region_name=region)

    rows = fetch(session.client("logs"), cfg["aws"]["audit_log_group"], features, args.minutes,
                 args.request_id_prefix)
    n = len(rows)
    log(f"{n} unique predictions in the last {args.minutes} min"
        + (f" with request_id prefix {args.request_id_prefix!r}" if args.request_id_prefix else ""))
    if n >= INSIGHTS_LIMIT:
        log(f"WARNING: hit the {INSIGHTS_LIMIT}-row Insights limit; PSI uses the newest rows only")
    if n < args.min_samples:
        raise SystemExit(f"only {n} rows (< --min-samples {args.min_samples}); not publishing PSI on so little data. "
                         "Audit lines reach CloudWatch a few minutes after scoring, so retry shortly.")

    values = {f: np.array([float(r[f]) for r in rows if f in r]) for f in features}
    values["fraud_score"] = np.array([float(r["fraud_score"]) for r in rows if "fraud_score" in r])
    specs = {**baseline["features"], "fraud_score": baseline["score"]}
    result = {name: drift.feature_psi(specs[name], v) for name, v in values.items()}
    max_feature_psi = max(v for k, v in result.items() if k != "fraud_score")

    print(f"\n{'feature':26s}{'PSI':>8s}  status")
    for name, v in sorted(result.items(), key=lambda kv: -kv[1]):
        print(f"{name:26s}{v:8.3f}  {next(label for cut, label in STATUS if v >= cut)}")
    print()

    if args.dry_run:
        log("dry run: nothing published")
        return
    data = [{"MetricName": "PSI", "Dimensions": [{"Name": "Feature", "Value": k}], "Value": v, "Unit": "None"}
            for k, v in result.items()]
    data += [{"MetricName": "MaxFeaturePSI", "Value": max_feature_psi, "Unit": "None"},
             {"MetricName": "DriftSampleSize", "Value": float(n), "Unit": "Count"}]
    session.client("cloudwatch").put_metric_data(Namespace=NS, MetricData=data)
    log(f"published {len(data)} metrics; MaxFeaturePSI={max_feature_psi:.3f}")


if __name__ == "__main__":
    main()
