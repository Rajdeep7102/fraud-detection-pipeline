"""Features checked against a hand-built history for one card (plus a second card for isolation)."""
import duckdb
import pandas as pd
import pytest

from fraud.features import feature_names, feature_sql

# card 1, home US. Columns: id, ts, amount, merchant, device, country
CARD1 = [
    (1, "2026-06-10 12:00:00", 10.0, 7, 100, "US"),
    (2, "2026-06-10 12:30:00", 20.0, 7, 100, "US"),    # 30 min after #1
    (3, "2026-06-10 13:10:00", 30.0, 8, 101, "FR"),    # new merchant, new device, abroad
    (4, "2026-06-12 12:00:00", 60.0, 7, 100, "US"),    # 2 days later
]


@pytest.fixture(scope="module")
def feats():
    rows = [(i, 1, ts, a, m, d, c, 0, "normal") for i, ts, a, m, d, c in CARD1]
    rows.append((9, 2, "2026-06-10 12:05:00", 999.0, 7, 500, "CA", 1, "stealth"))  # other card, same merchant
    df = pd.DataFrame(rows, columns=["transaction_id", "card_id", "ts", "amount", "merchant_id",
                                     "device_id", "country", "is_fraud", "kind"])
    df["ts"] = pd.to_datetime(df["ts"])
    con = duckdb.connect()
    con.register("txn", df)
    con.register("cards", pd.DataFrame({"card_id": [1, 2], "home_country": ["US", "CA"]}))
    return con.execute(feature_sql()).df().set_index("transaction_id").sort_index()


def test_columns_match_spec(feats):
    assert [c for c in feats.columns if c in feature_names()] == feature_names()
    assert len(feature_names()) == 12


def test_velocity(feats):
    assert feats.loc[[1, 2, 3, 4], "txn_count_1h"].tolist() == [1, 2, 2, 1]   # #3: #2 is 40 min earlier, #1 is 70
    assert feats.loc[[1, 2, 3, 4], "txn_count_24h"].tolist() == [1, 2, 3, 1]
    assert feats.loc[[1, 2, 3, 4], "amount_sum_24h"].tolist() == [10.0, 30.0, 60.0, 60.0]
    assert feats.loc[1, "secs_since_last_txn"] == 2592000.0          # first txn default
    assert feats.loc[2, "secs_since_last_txn"] == 1800.0
    assert feats.loc[3, "secs_since_last_txn"] == 2400.0
    assert feats.loc[4, "secs_since_last_txn"] == 2 * 86400 - 70 * 60
    assert feats.loc[[1, 2, 3, 4], "distinct_merchants_24h"].tolist() == [1, 1, 2, 1]
    assert feats.loc[[1, 2, 3, 4], "distinct_countries_7d"].tolist() == [1, 1, 2, 2]


def test_amount_baseline_excludes_current_row(feats):
    assert feats.loc[1, "amount_to_avg_30d"] == 1.0 and feats.loc[1, "amount_zscore_30d"] == 0.0   # no history
    assert feats.loc[2, "amount_to_avg_30d"] == pytest.approx(20 / 10)
    assert feats.loc[3, "amount_to_avg_30d"] == pytest.approx(30 / 15)
    assert feats.loc[3, "amount_zscore_30d"] == pytest.approx((30 - 15) / pd.Series([10, 20]).std())
    assert feats.loc[4, "amount_to_avg_30d"] == pytest.approx(60 / 20)


def test_novelty_and_geography(feats):
    assert feats.loc[[1, 2, 3, 4], "is_new_merchant"].tolist() == [1, 0, 1, 0]
    assert feats.loc[[1, 2, 3, 4], "is_new_device"].tolist() == [1, 0, 1, 0]
    assert feats.loc[[1, 2, 3, 4], "is_foreign_country"].tolist() == [0, 0, 1, 0]


def test_hour_deviation(feats):
    assert feats.loc[1, "hour_deviation"] == 0.0                    # no history
    assert feats.loc[2, "hour_deviation"] == pytest.approx(0.5)     # 12:30 vs usual 12:00
    mu = _circ_mean([12.0, 12.5, 13 + 10 / 60])                     # usual hour from the 3 prior txns
    assert feats.loc[4, "hour_deviation"] == pytest.approx(min(abs(12.0 - mu), 24 - abs(12.0 - mu)), abs=1e-6)


def test_cards_are_isolated(feats):
    assert feats.loc[9, "txn_count_24h"] == 1                       # card 1's rows never leak in
    assert feats.loc[9, "is_new_merchant"] == 1                     # merchant 7 is new for card 2


def _circ_mean(hours):
    import math
    s = sum(math.sin(2 * math.pi * h / 24) for h in hours) / len(hours)
    c = sum(math.cos(2 * math.pi * h / 24) for h in hours) / len(hours)
    return (math.atan2(s, c) * 24 / (2 * math.pi) + 24) % 24
