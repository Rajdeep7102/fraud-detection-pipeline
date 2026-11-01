"""Serving contract: the container's scores must equal the training-side model's, every prediction must be
audit-logged as one JSON line, and bad input must be rejected (and logged) rather than scored."""
import json
import sys

import duckdb
import joblib
import numpy as np
import pytest

from fraud.config import ROOT, load_config, model_dir
from fraud.features import feature_names

ARTIFACTS = model_dir()
pytestmark = pytest.mark.skipif(not (ARTIFACTS / "model.joblib").exists(), reason="train a model first")

pytest.importorskip("flask", reason="serving tests need the full requirements.txt")
sys.path.insert(0, str(ROOT / "serving"))
from app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return create_app(str(ARTIFACTS)).test_client()


@pytest.fixture(scope="module")
def rows():
    cfg = load_config()
    if not (ROOT / cfg["features"]["dir"] / "bucket=00").exists():
        pytest.skip("needs the generated feature data (python -m fraud.data.generate, then fraud.features.build)")
    glob = (ROOT / cfg["features"]["dir"] / "bucket=00" / "*.parquet").as_posix()
    cols = ", ".join(feature_names())
    df = duckdb.sql(f"""
        (SELECT transaction_id, {cols} FROM '{glob}' WHERE day >= 70 AND is_fraud = 1 LIMIT 40)
        UNION ALL
        (SELECT transaction_id, {cols} FROM '{glob}' WHERE day >= 70 AND is_fraud = 0 LIMIT 60)""").df()
    return df


def _instances(df):
    return [{k: (int(v) if k == "transaction_id" else float(v)) for k, v in r.items()}
            for r in df.to_dict("records")]


def _audit_lines(out: str):
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def test_ping(client):
    r = client.get("/ping")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_scores_match_training_side_model(client, rows, capsys):
    resp = client.post("/invocations", json={"instances": _instances(rows)})
    assert resp.status_code == 200
    body = resp.get_json()

    model = joblib.load(ARTIFACTS / "model.joblib")
    expected = model.predict_proba(rows[feature_names()].to_numpy(np.float32))[:, 1]
    got = np.array([p["fraud_score"] for p in body["predictions"]])
    assert np.allclose(got, expected, atol=1e-6)            # fraud_score is rounded to 6 decimals

    meta = json.loads((ARTIFACTS / "model_meta.json").read_text())
    for pred, score in zip(body["predictions"], expected):
        assert pred["decision"] == ("FRAUD" if score >= meta["threshold"] else "LEGIT")
        assert pred["confidence"] == pytest.approx(max(score, 1 - score), abs=1e-6)

    audit = _audit_lines(capsys.readouterr().out)
    assert len(audit) == len(rows)                           # one line per prediction
    first = audit[0]
    assert {"event", "ts", "request_id", "transaction_id", "model_version", "threshold", "fraud_score",
            "confidence", "decision", "features", "latency_ms"} <= set(first)
    assert list(first["features"]) == feature_names()
    assert len({a["request_id"] for a in audit}) == 1       # one request id for the whole batch


@pytest.mark.parametrize("payload", [
    {},                                                      # no instances
    {"instances": []},
    {"instances": [{"txn_count_1h": 1}]},                    # missing features
    {"instances": ["not an object"]},
])
def test_rejects_bad_payloads(client, payload, capsys):
    r = client.post("/invocations", json=payload)
    assert r.status_code == 400 and "error" in r.get_json()
    audit = _audit_lines(capsys.readouterr().out)
    assert [a["event"] for a in audit] == ["request_rejected"]


def test_rejects_non_finite_and_wrong_types(client, rows):
    inst = _instances(rows.head(1))
    inst[0]["amount_sum_24h"] = float("nan")
    assert client.post("/invocations", json={"instances": inst}).status_code == 400
    inst[0]["amount_sum_24h"] = "12"
    assert client.post("/invocations", json={"instances": inst}).status_code == 400


def test_rejects_malformed_json_and_oversize_batch(client, rows):
    assert client.post("/invocations", data="{oops", content_type="application/json").status_code == 400
    big = _instances(rows.head(1)) * 101
    assert client.post("/invocations", json={"instances": big}).status_code == 400
