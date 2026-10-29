"""Plain-English names for the model's inputs, shared by the manager report and the demo app."""

FEATURES = {
    "txn_count_1h": ("Purchases in the last hour", "Many purchases in a short time is the classic sign of card testing."),
    "txn_count_24h": ("Purchases in the last 24 hours", "Sustained unusual activity over a day."),
    "amount_sum_24h": ("Total spent in the last 24 hours", "A large total can mean a cash-out after a takeover."),
    "secs_since_last_txn": ("Seconds since the previous purchase", "Very short gaps point to scripted, rapid-fire use."),
    "distinct_merchants_24h": ("Different merchants in 24 hours", "Fraudsters often try many merchants quickly."),
    "distinct_countries_7d": ("Different countries in 7 days", "A card appearing in several countries in a week."),
    "amount_to_avg_30d": ("Amount vs the card's 30-day average", "How big this purchase is compared with this customer's normal."),
    "amount_zscore_30d": ("Amount vs the card's usual range", "Same idea, adjusted for how much the customer's spending normally varies."),
    "is_new_merchant": ("First time at this merchant", "Customers mostly shop where they have shopped before."),
    "is_new_device": ("First time on this device", "A new phone or browser is a common sign of account takeover."),
    "is_foreign_country": ("Purchase outside the home country", "Cross-border use, which is also common for genuine travel."),
    "hour_deviation": ("Hours away from the usual time of day", "Purchases at unusual hours for this customer."),
}

KINDS = {
    "card_testing": ("Card testing", "A burst of tiny purchases at many merchants, then a larger one."),
    "account_takeover": ("Account takeover", "A new device, often abroad, with several large purchases."),
    "stealth": ("Stealth fraud", "A few moderately large purchases that look almost normal."),
}


def label(feature: str) -> str:
    return FEATURES[feature][0]

# Short names for chart axes, where the long labels get truncated.
SHORT = {
    "txn_count_1h": "Purchases, 1h", "txn_count_24h": "Purchases, 24h", "amount_sum_24h": "Spend, 24h",
    "secs_since_last_txn": "Gap since last", "distinct_merchants_24h": "Merchants, 24h",
    "distinct_countries_7d": "Countries, 7d", "amount_to_avg_30d": "Amount vs average",
    "amount_zscore_30d": "Amount vs range", "is_new_merchant": "New merchant", "is_new_device": "New device",
    "is_foreign_country": "Foreign country", "hour_deviation": "Odd hour",
}


def short(feature: str) -> str:
    return "Fraud score" if feature == "fraud_score" else SHORT[feature]
