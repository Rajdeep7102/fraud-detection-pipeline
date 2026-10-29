"""Live model health from CloudWatch for the last N hours. Every function degrades to None / [] when AWS
is unreachable, so a report can still be generated offline."""
from __future__ import annotations

import datetime as dt

import boto3

from fraud.config import load_config
from fraud.features import feature_names

NS = "FraudDetection"
ALARMS = {
    "fraud-flag-rate-high": "Flag rate unusually high",
    "fraud-low-confidence": "Model confidence low",
    "fraud-feature-drift": "Input data drift",
    "fraud-score-drift": "Score distribution drift",
    "fraud-endpoint-5xx": "Endpoint errors",
}


def _session():
    return boto3.Session(region_name=load_config()["aws"]["region"])


def _window(hours: int):
    end = dt.datetime.now(dt.timezone.utc)
    return end - dt.timedelta(hours=hours), end


def health(hours: int = 24) -> dict | None:
    """Predictions, flag rate, confidence and latency over the window; None if CloudWatch cannot be read."""
    try:
        cw = _session().client("cloudwatch")
        cfg = load_config()["aws"]
        start, end = _window(hours)
        period = max(60, int(hours * 3600) // 60 * 60)
        period = min(period, 86400)
        ep = [{"Name": "EndpointName", "Value": cfg["endpoint_name"]}, {"Name": "VariantName", "Value": "AllTraffic"}]

        def q(i, ns, name, stat, dims=()):
            return {"Id": i, "MetricStat": {"Metric": {"Namespace": ns, "MetricName": name, "Dimensions": list(dims)},
                                            "Period": period, "Stat": stat}, "ReturnData": True}

        res = cw.get_metric_data(StartTime=start, EndTime=end, MetricDataQueries=[
            q("pred", NS, "PredictionCount", "Sum"), q("flag", NS, "FraudFlagCount", "Sum"),
            q("conf", NS, "ModelConfidence", "Average"), q("conf10", NS, "ModelConfidence", "p10"),
            q("lat", "AWS/SageMaker", "ModelLatency", "Average", ep), q("err", "AWS/SageMaker", "Invocation5XXErrors", "Sum", ep),
            q("rej", NS, "RequestRejectedCount", "Sum")])
        vals = {r["Id"]: (sum(r["Values"]) if r["Id"] in ("pred", "flag", "err", "rej") else
                          (sum(r["Values"]) / len(r["Values"]) if r["Values"] else None))
                for r in res["MetricDataResults"]}
        pred = vals.get("pred") or 0.0
        return {"hours": hours, "predictions": int(pred), "flagged": int(vals.get("flag") or 0),
                "flag_rate_pct": 100 * (vals.get("flag") or 0) / pred if pred else None,
                "confidence_avg": vals.get("conf"), "confidence_p10": vals.get("conf10"),
                "latency_ms": (vals["lat"] / 1000) if vals.get("lat") is not None else None,
                "errors_5xx": int(vals.get("err") or 0), "rejected": int(vals.get("rej") or 0)}
    except Exception:  # noqa: BLE001 - offline / no credentials / no permissions: caller shows "unavailable"
        return None


def alarms() -> list[dict]:
    try:
        resp = _session().client("cloudwatch").describe_alarms(AlarmNames=list(ALARMS))["MetricAlarms"]
        order = {n: i for i, n in enumerate(ALARMS)}
        return sorted(({"name": a["AlarmName"], "label": ALARMS[a["AlarmName"]], "state": a["StateValue"],
                        "changed": a["StateUpdatedTimestamp"]} for a in resp), key=lambda a: order[a["name"]])
    except Exception:  # noqa: BLE001
        return []


def psi_by_feature(hours: int = 24) -> dict[str, float]:
    """Highest PSI published per feature (and fraud_score) in the window, from the drift job."""
    try:
        cw = _session().client("cloudwatch")
        start, end = _window(hours)
        names = feature_names() + ["fraud_score"]
        queries = [{"Id": f"f{i}", "Label": n, "ReturnData": True, "MetricStat": {
            "Metric": {"Namespace": NS, "MetricName": "PSI", "Dimensions": [{"Name": "Feature", "Value": n}]},
            "Period": min(max(hours * 3600 // 60 * 60, 60), 86400), "Stat": "Maximum"}} for i, n in enumerate(names)]
        out = {}
        for r in cw.get_metric_data(StartTime=start, EndTime=end, MetricDataQueries=queries)["MetricDataResults"]:
            if r["Values"]:
                out[r["Label"]] = max(r["Values"])
        return out
    except Exception:  # noqa: BLE001
        return {}
