"""Creates (or updates) the CloudWatch monitoring for the fraud endpoint. Idempotent.

  * log retention on the endpoint log group
  * metric filters that turn JSON audit lines into FraudDetection metrics
  * an SNS topic + alarms (flag rate, confidence, feature drift, score drift, 5XX)
  * the FraudDetection dashboard (also written to monitoring/dashboard.json)

    python monitoring/setup_monitoring.py
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3
import yaml

ROOT = Path(__file__).resolve().parents[1]
NS = "FraudDetection"
RETENTION_DAYS = 30
PSI_ALARM = 0.2
LOW_CONFIDENCE = 0.6
MIN_REQUESTS_FOR_RATE = 500     # flag-rate alarm ignores 5-minute windows with fewer predictions


def log(msg: str) -> None:
    print(f"[monitoring] {msg}", flush=True)


def metric_filters() -> list[dict]:
    """name, pattern, metric name, value, default. Patterns use CloudWatch's JSON filter syntax."""
    pred = '$.event = "prediction"'
    return [
        {"name": "fraud-prediction-count", "pattern": f"{{ {pred} }}", "metric": "PredictionCount", "value": "1", "default": 0},
        {"name": "fraud-model-confidence", "pattern": f"{{ {pred} }}", "metric": "ModelConfidence", "value": "$.confidence"},
        {"name": "fraud-score", "pattern": f"{{ {pred} }}", "metric": "FraudScore", "value": "$.fraud_score"},
        {"name": "fraud-scoring-latency", "pattern": f"{{ {pred} }}", "metric": "ScoringLatencyMs", "value": "$.latency_ms"},
        {"name": "fraud-flag-count", "pattern": f'{{ {pred} && $.decision = "FRAUD" }}', "metric": "FraudFlagCount",
         "value": "1", "default": 0},
        {"name": "fraud-low-confidence-count", "pattern": f"{{ {pred} && $.confidence < {LOW_CONFIDENCE} }}",
         "metric": "LowConfidenceCount", "value": "1", "default": 0},
        {"name": "fraud-request-rejected-count", "pattern": '{ $.event = "request_rejected" }',
         "metric": "RequestRejectedCount", "value": "1", "default": 0},
    ]


def stat(namespace: str, name: str, dims: list[dict], period: int, st: str) -> dict:
    return {"Metric": {"Namespace": namespace, "MetricName": name, "Dimensions": dims}, "Period": period, "Stat": st}


def build_alarms(endpoint: str, flag_rate_threshold: float, topic_arn: str) -> list[dict]:
    ep = [{"Name": "EndpointName", "Value": endpoint}, {"Name": "VariantName", "Value": "AllTraffic"}]
    common = {"AlarmActions": [topic_arn], "TreatMissingData": "notBreaching"}
    return [
        {"AlarmName": "fraud-flag-rate-high", "AlarmDescription":
            f"Share of predictions flagged FRAUD is above {flag_rate_threshold:.2%} (3x the {flag_rate_threshold / 3:.2%} "
            f"measured on the held-out test set), over >= {MIN_REQUESTS_FOR_RATE} predictions.",
         "Metrics": [
             {"Id": "m1", "MetricStat": stat(NS, "FraudFlagCount", [], 300, "Sum"), "ReturnData": False},
             {"Id": "m2", "MetricStat": stat(NS, "PredictionCount", [], 300, "Sum"), "ReturnData": False},
             {"Id": "e1", "Expression": f"IF(m2 >= {MIN_REQUESTS_FOR_RATE}, m1 / m2, 0)", "Label": "flag rate", "ReturnData": True}],
         "ComparisonOperator": "GreaterThanThreshold", "Threshold": flag_rate_threshold,
         "EvaluationPeriods": 1, "DatapointsToAlarm": 1, **common},
        {"AlarmName": "fraud-low-confidence", "AlarmDescription":
            f"p10 of model confidence below {LOW_CONFIDENCE} in 3 of 5 five-minute periods.",
         "Namespace": NS, "MetricName": "ModelConfidence", "ExtendedStatistic": "p10", "Period": 300,
         "ComparisonOperator": "LessThanThreshold", "Threshold": LOW_CONFIDENCE,
         "EvaluationPeriods": 5, "DatapointsToAlarm": 3, **common},
        {"AlarmName": "fraud-feature-drift", "AlarmDescription":
            f"Largest per-feature PSI from the drift job is above {PSI_ALARM}.",
         "Namespace": NS, "MetricName": "MaxFeaturePSI", "Statistic": "Maximum", "Period": 900,
         "ComparisonOperator": "GreaterThanThreshold", "Threshold": PSI_ALARM,
         "EvaluationPeriods": 1, "DatapointsToAlarm": 1, **common},
        {"AlarmName": "fraud-score-drift", "AlarmDescription":
            f"PSI of the fraud_score distribution is above {PSI_ALARM}.",
         "Namespace": NS, "MetricName": "PSI", "Dimensions": [{"Name": "Feature", "Value": "fraud_score"}],
         "Statistic": "Maximum", "Period": 900,
         "ComparisonOperator": "GreaterThanThreshold", "Threshold": PSI_ALARM,
         "EvaluationPeriods": 1, "DatapointsToAlarm": 1, **common},
        {"AlarmName": "fraud-endpoint-5xx", "AlarmDescription": "The endpoint returned server errors.",
         "Namespace": "AWS/SageMaker", "MetricName": "Invocation5XXErrors", "Dimensions": ep,
         "Statistic": "Sum", "Period": 300,
         "ComparisonOperator": "GreaterThanThreshold", "Threshold": 0,
         "EvaluationPeriods": 1, "DatapointsToAlarm": 1, **common},
    ]


def build_dashboard(endpoint: str, region: str, features: list[str], flag_rate_pct: float, group: str) -> dict:
    ep = ["EndpointName", endpoint, "VariantName", "AllTraffic"]
    fd = lambda name, opts: [NS, name, opts]  # noqa: E731

    def w(x, y, width, height, title, metrics, extra=None, view="timeSeries"):
        props = {"title": title, "region": region, "metrics": metrics, "view": view, "stacked": False, "period": 300,
                 "stat": "Average", **(extra or {})}
        return {"type": "metric", "x": x, "y": y, "width": width, "height": height, "properties": props}

    psi_metrics = [[NS, "PSI", "Feature", f, {"stat": "Maximum", "label": f}] for f in features + ["fraud_score"]]
    return {"widgets": [
        {"type": "text", "x": 0, "y": 0, "width": 24, "height": 2, "properties": {"markdown":
            f"## Fraud detection endpoint `{endpoint}`\nTraffic and health (row 1), model behaviour from the JSON audit log "
            f"(row 2), drift from the PSI job (row 3), recent flagged transactions (row 4). Audit-log metrics lag "
            f"a few minutes; PSI points appear when `monitoring/drift_job.py` runs."}},
        w(0, 2, 8, 6, "Invocations", [["AWS/SageMaker", "Invocations", *ep, {"stat": "Sum"}]]),
        w(8, 2, 8, 6, "Model latency (ms)", [
            ["AWS/SageMaker", "ModelLatency", *ep, {"id": "m1", "stat": "Average", "visible": False}],
            ["AWS/SageMaker", "ModelLatency", *ep, {"id": "m2", "stat": "p99", "visible": False}],
            [{"expression": "m1 / 1000", "label": "average", "id": "e1"}],
            [{"expression": "m2 / 1000", "label": "p99", "id": "e2"}]]),
        w(16, 2, 8, 6, "Errors", [
            ["AWS/SageMaker", "Invocation5XXErrors", *ep, {"stat": "Sum", "label": "5XX"}],
            fd("RequestRejectedCount", {"stat": "Sum", "label": "rejected (400)"})]),
        w(0, 8, 8, 6, "Model confidence", [
            fd("ModelConfidence", {"stat": "Average", "label": "average"}),
            fd("ModelConfidence", {"stat": "p10", "label": "p10"})],
          {"annotations": {"horizontal": [{"label": "alarm", "value": LOW_CONFIDENCE}]}, "yAxis": {"left": {"min": 0.5, "max": 1}}}),
        w(8, 8, 8, 6, "Flagged as FRAUD (%)", [
            fd("FraudFlagCount", {"id": "m1", "stat": "Sum", "visible": False}),
            fd("PredictionCount", {"id": "m2", "stat": "Sum", "visible": False}),
            [{"expression": "IF(m2 > 0, 100 * m1 / m2, 0)", "label": "% flagged", "id": "e1"}]],
          {"annotations": {"horizontal": [{"label": "alarm (3x test-set rate)", "value": round(flag_rate_pct, 3)}]}}),
        w(16, 8, 8, 6, "Predictions and low-confidence", [
            fd("PredictionCount", {"stat": "Sum", "label": "predictions"}),
            fd("LowConfidenceCount", {"stat": "Sum", "label": f"confidence < {LOW_CONFIDENCE}"})]),
        w(0, 14, 18, 7, "Drift: PSI per feature (alarm at 0.2)", psi_metrics,
          {"annotations": {"horizontal": [{"label": "significant shift", "value": PSI_ALARM}]}, "period": 900}),
        w(18, 14, 6, 7, "Largest feature PSI", [fd("MaxFeaturePSI", {"stat": "Maximum"}),
                                                  fd("DriftSampleSize", {"stat": "Maximum", "label": "rows scored"})],
          {"period": 900}, view="singleValue"),
        {"type": "log", "x": 0, "y": 21, "width": 24, "height": 7, "properties": {
            "title": "FRAUD decisions, newest first (deduplicated)", "region": region, "view": "table",
            "query": f"SOURCE '{group}' | fields @timestamp, transaction_id, fraud_score, confidence, request_id"
                     f" | filter event = \"prediction\" and decision = \"FRAUD\""
                     f" | sort @timestamp desc | dedup request_id, transaction_id"}},
    ]}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    region, endpoint = cfg["aws"]["region"], cfg["aws"]["endpoint_name"]
    group = cfg["aws"]["audit_log_group"]                      # audit records, written by the container
    stdout_group = f"/aws/sagemaker/Endpoints/{endpoint}"      # gunicorn/platform output only
    features = [f["name"] for f in yaml.safe_load(
        (ROOT / "src/fraud/features/feature_spec.yaml").read_text())["features"]]
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())["confusion"]
    test_rows = sum(m.values())
    flag_rate = (m["tp"] + m["fp"]) / test_rows
    session = boto3.Session(region_name=region)
    logs, cw, sns = session.client("logs"), session.client("cloudwatch"), session.client("sns")

    try:
        logs.create_log_group(logGroupName=group)
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    for g in (group, stdout_group):
        logs.put_retention_policy(logGroupName=g, retentionInDays=RETENTION_DAYS)
    log(f"retention {RETENTION_DAYS} days on {group} and {stdout_group}")
    # Earlier versions counted audit lines scraped from stdout (lossy); remove those filters so nothing double-counts.
    for f in metric_filters():
        try:
            logs.delete_metric_filter(logGroupName=stdout_group, filterName=f["name"])
        except logs.exceptions.ResourceNotFoundException:
            pass

    for f in metric_filters():
        transformation = {"metricName": f["metric"], "metricNamespace": NS, "metricValue": f["value"]}
        if "default" in f:
            transformation["defaultValue"] = f["default"]
        logs.put_metric_filter(logGroupName=group, filterName=f["name"], filterPattern=f["pattern"],
                               metricTransformations=[transformation])
    log(f"{len(metric_filters())} metric filters -> namespace {NS}")

    topic_arn = sns.create_topic(Name="fraud-alerts")["TopicArn"]
    for a in build_alarms(endpoint, flag_rate_threshold=3 * flag_rate, topic_arn=topic_arn):
        cw.put_metric_alarm(**a)
    log(f"5 alarms -> {topic_arn} (flag-rate threshold {3 * flag_rate:.3%}, from test-set rate {flag_rate:.3%})")

    body = build_dashboard(endpoint, region, features, 100 * 3 * flag_rate, group)
    (ROOT / "monitoring" / "dashboard.json").write_text(json.dumps(body, indent=2))
    res = cw.put_dashboard(DashboardName="FraudDetection", DashboardBody=json.dumps(body))
    if res.get("DashboardValidationMessages"):
        raise SystemExit(f"dashboard rejected: {res['DashboardValidationMessages']}")
    log(f"dashboard: https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name=FraudDetection")
    log(f"to get alarm emails: aws sns subscribe --topic-arn {topic_arn} --protocol email --notification-endpoint <you@example.com>")


if __name__ == "__main__":
    main()
