"""Manager-facing KPIs derived from reports/metrics.json (the held-out test evaluation) and the saved
flagged/fraud rows. Pure functions, no AWS."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

from fraud.config import ROOT, load_config
from fraud.reporting.glossary import KINDS

DEFAULT_REVIEW_COST = 5.0   # ASSUMPTION: analyst cost to review one alert, in dollars; shown as an editable assumption


def load_metrics(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "reports" / "metrics.json").read_text())


def headline(m: dict) -> dict:
    c = m["confusion"]
    rows = m["split"]["rows"]
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {
        "rows": rows,
        "fraud_rows": m["split"]["fraud_rows"],
        "fraud_caught_pct": 100 * m["recall"],
        "alert_precision_pct": 100 * m["precision"],
        "caught": tp, "missed": fn, "false_alarms": fp,
        "alerts": tp + fp,
        "alerts_per_10k": 10_000 * (tp + fp) / rows,
        "false_alarms_per_10k": 10_000 * fp / rows,
        "real_per_100_alerts": 100 * tp / max(tp + fp, 1),
        "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["f1"],
        "recall_ci": [100 * x for x in m["recall_95ci"]],
        "precision_ci": [100 * x for x in m["precision_95ci"]],
        "model_version": m["model_version"],
        "test_days": m["split"]["test_days"],
    }


def by_type(m: dict) -> list[dict]:
    out = []
    for key, (name, blurb) in KINDS.items():
        k = m["by_kind"][key]
        out.append({"key": key, "name": name, "blurb": blurb, "rows": k["rows"], "caught": k["flagged"],
                    "recall_pct": 100 * k["recall"], "share_of_fraud_pct": 100 * k["rows"] / m["split"]["fraud_rows"]})
    return out


def dollars(review_cost: float = DEFAULT_REVIEW_COST) -> dict | None:
    """Value of fraud caught/missed using the dataset's own (synthetic) transaction amounts.
    Returns None if the flagged-rows file or raw data is missing."""
    cfg = load_config()
    flagged = ROOT / "reports" / "test_flagged_or_fraud.parquet"
    raw = ROOT / cfg["data"]["raw_dir"] / "transactions"
    if not flagged.exists() or not raw.exists():
        return None
    q = f"""
        SELECT f.is_fraud, f.flagged, count(*) AS n, sum(t.amount) AS amount
        FROM '{flagged.as_posix()}' f JOIN '{(raw / "*" / "*.parquet").as_posix()}' t USING (transaction_id)
        GROUP BY ALL"""
    rows = {(bool(r[0]), bool(r[1])): (r[2], r[3]) for r in duckdb.sql(q).fetchall()}
    caught = rows.get((True, True), (0, 0.0))
    missed = rows.get((True, False), (0, 0.0))
    false_alarm = rows.get((False, True), (0, 0.0))
    total_fraud = caught[1] + missed[1]
    review = (caught[0] + false_alarm[0]) * review_cost
    return {"fraud_total": total_fraud, "fraud_caught": caught[1], "fraud_missed": missed[1],
            "share_caught_pct": 100 * caught[1] / total_fraud if total_fraud else 0.0,
            "alerts": caught[0] + false_alarm[0], "review_cost_per_alert": review_cost, "review_cost": review,
            "net_benefit": caught[1] - review}
