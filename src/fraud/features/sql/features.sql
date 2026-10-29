-- The 12 engineered features. Expects two relations:
--   txn(transaction_id, card_id, ts, amount, merchant_id, device_id, country, is_fraud, kind)
--   cards(card_id, home_country)
-- Every window is per card and only looks backwards (plus the current row), so a feature
-- value never depends on a later transaction. 'Prior' windows exclude the current row.
-- Defaults for a card's first transaction are documented in feature_spec.yaml.

WITH base AS (
    SELECT
        t.transaction_id, t.card_id, t.ts, t.amount, t.merchant_id, t.device_id, t.country,
        t.is_fraud, t.kind, c.home_country,
        date_diff('day', TIMESTAMP '2026-06-01', t.ts)               AS day,
        hour(t.ts) + minute(t.ts) / 60.0                              AS hod,
        sin(2 * pi() * (hour(t.ts) + minute(t.ts) / 60.0) / 24)       AS hod_sin,
        cos(2 * pi() * (hour(t.ts) + minute(t.ts) / 60.0) / 24)       AS hod_cos
    FROM txn t
    JOIN cards c USING (card_id)
),
w AS (
    SELECT
        *,
        -- velocity ------------------------------------------------------------------------
        COUNT(*)    OVER h1                                          AS txn_count_1h,
        COUNT(*)    OVER h24                                         AS txn_count_24h,
        SUM(amount) OVER h24                                         AS amount_sum_24h,
        epoch(ts) - epoch(LAG(ts) OVER seq)                          AS secs_since_last_raw,
        COUNT(DISTINCT merchant_id) OVER h24                         AS distinct_merchants_24h,
        COUNT(DISTINCT country)     OVER d7                          AS distinct_countries_7d,
        -- behavioural baselines over the prior 30 days (current row excluded) --------------
        AVG(amount)         OVER d30x                                AS avg_amt_30d,
        STDDEV_SAMP(amount) OVER d30x                                AS sd_amt_30d,
        AVG(hod_sin)        OVER d30x                                AS mean_sin_30d,
        AVG(hod_cos)        OVER d30x                                AS mean_cos_30d,
        ROW_NUMBER() OVER (PARTITION BY card_id, merchant_id ORDER BY ts, transaction_id) AS merchant_seen_n,
        ROW_NUMBER() OVER (PARTITION BY card_id, device_id   ORDER BY ts, transaction_id) AS device_seen_n
    FROM base
    WINDOW
        seq  AS (PARTITION BY card_id ORDER BY ts, transaction_id),
        h1   AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 1 HOUR  PRECEDING AND CURRENT ROW),
        h24  AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW),
        d7   AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 7 DAY   PRECEDING AND CURRENT ROW),
        d30x AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 30 DAY  PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW)
)
SELECT
    transaction_id, card_id, ts, day, is_fraud, kind,
    txn_count_1h::INTEGER                                                       AS txn_count_1h,
    txn_count_24h::INTEGER                                                      AS txn_count_24h,
    amount_sum_24h::DOUBLE                                                      AS amount_sum_24h,
    COALESCE(secs_since_last_raw, 2592000)::DOUBLE                              AS secs_since_last_txn,
    distinct_merchants_24h::INTEGER                                             AS distinct_merchants_24h,
    distinct_countries_7d::INTEGER                                              AS distinct_countries_7d,
    COALESCE(amount / NULLIF(avg_amt_30d, 0), 1.0)::DOUBLE                      AS amount_to_avg_30d,
    COALESCE((amount - avg_amt_30d) / NULLIF(sd_amt_30d, 0), 0.0)::DOUBLE       AS amount_zscore_30d,
    (merchant_seen_n = 1)::TINYINT                                              AS is_new_merchant,
    (device_seen_n = 1)::TINYINT                                                AS is_new_device,
    (country <> home_country)::TINYINT                                          AS is_foreign_country,
    -- circular distance (hours) between this hour of day and the card's usual hour
    COALESCE(
        LEAST(
            abs(hod - (atan2(mean_sin_30d, mean_cos_30d) * 24 / (2 * pi()) + 24) % 24),
            24 - abs(hod - (atan2(mean_sin_30d, mean_cos_30d) * 24 / (2 * pi()) + 24) % 24)
        ), 0.0)::DOUBLE                                                         AS hour_deviation
FROM w
