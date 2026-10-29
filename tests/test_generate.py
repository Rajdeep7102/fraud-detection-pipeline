import duckdb

from fraud.config import load_config
from fraud.data.generate import generate


def test_small_generation(tmp_path):
    cfg = load_config()
    summary = generate(cfg, tmp_path / "raw", scale=0.002)
    glob = str(tmp_path / "raw" / "transactions" / "*" / "*.parquet").replace("\\", "/")
    rows, ids, fraud, days = duckdb.sql(
        f"select count(*), count(distinct transaction_id), sum(is_fraud), "
        f"date_diff('day', min(ts), max(ts)) from '{glob}'"
    ).fetchone()

    assert rows == summary["rows"] == int(cfg["data"]["n_transactions"] * 0.002)
    assert ids == rows
    assert fraud == summary["fraud_rows"] > 0
    assert days < cfg["data"]["days"]
    assert {"card_testing", "account_takeover", "stealth", "travel"} <= set(summary["episodes_by_kind"])


def test_deterministic(tmp_path):
    cfg = load_config()
    a = generate(cfg, tmp_path / "a", scale=0.001)
    b = generate(cfg, tmp_path / "b", scale=0.001)
    a.pop("seconds"), b.pop("seconds")
    assert a == b
