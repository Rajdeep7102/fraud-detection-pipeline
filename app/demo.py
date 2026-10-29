"""Interactive demo: score a transaction on the live SageMaker endpoint, see why, and read the model's
performance, live monitoring and a downloadable manager report.

    streamlit run app/demo.py
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from fraud.config import ROOT, load_config
from fraud.features import feature_names
from fraud.reporting import kpis, live, report
from fraud.reporting.glossary import FEATURES, label, short

CFG = load_config()
REGION, ENDPOINT = CFG["aws"]["region"], CFG["aws"]["endpoint_name"]
DASHBOARD_URL = f"https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#dashboards:name=FraudDetection"

SCENARIOS = {
    "Everyday purchase": dict(txn_count_1h=1, txn_count_24h=2, amount_sum_24h=60.0, secs_since_last_txn=40000.0,
                              distinct_merchants_24h=2, distinct_countries_7d=1, amount_to_avg_30d=1.0, amount_zscore_30d=0.1,
                              is_new_merchant=0, is_new_device=0, is_foreign_country=0, hour_deviation=1.0),
    "Card testing burst": dict(txn_count_1h=14, txn_count_24h=15, amount_sum_24h=90.0, secs_since_last_txn=45.0,
                               distinct_merchants_24h=13, distinct_countries_7d=1, amount_to_avg_30d=0.3, amount_zscore_30d=-0.4,
                               is_new_merchant=1, is_new_device=0, is_foreign_country=1, hour_deviation=2.0),
    "Account takeover": dict(txn_count_1h=3, txn_count_24h=4, amount_sum_24h=900.0, secs_since_last_txn=1200.0,
                             distinct_merchants_24h=3, distinct_countries_7d=2, amount_to_avg_30d=6.0, amount_zscore_30d=4.5,
                             is_new_merchant=1, is_new_device=1, is_foreign_country=1, hour_deviation=6.0),
    "Genuine holiday abroad": dict(txn_count_1h=2, txn_count_24h=5, amount_sum_24h=240.0, secs_since_last_txn=7200.0,
                                   distinct_merchants_24h=4, distinct_countries_7d=2, amount_to_avg_30d=1.8, amount_zscore_30d=1.2,
                                   is_new_merchant=1, is_new_device=0, is_foreign_country=1, hour_deviation=3.0),
    "Stealth fraud": dict(txn_count_1h=1, txn_count_24h=2, amount_sum_24h=180.0, secs_since_last_txn=30000.0,
                          distinct_merchants_24h=2, distinct_countries_7d=1, amount_to_avg_30d=2.5, amount_zscore_30d=1.5,
                          is_new_merchant=1, is_new_device=0, is_foreign_country=0, hour_deviation=1.5),
}
SCENARIO_NOTE = {
    "Everyday purchase": "A normal purchase at a familiar pattern.",
    "Card testing burst": "Many small purchases at many merchants within the hour.",
    "Account takeover": "A large purchase on a new device, abroad, well above the card's usual spend.",
    "Genuine holiday abroad": "A real customer travelling. It scores high but stays under the alert line; about 6% of such purchases are flagged in testing.",
    "Stealth fraud": "Looks almost normal. The model is known to miss this type, which is part of the honest picture.",
}
# Widget ranges: (min, max, step) for the numeric inputs; binary inputs are checkboxes.
RANGES = {"txn_count_1h": (0, 40, 1), "txn_count_24h": (0, 80, 1), "amount_sum_24h": (0.0, 5000.0, 10.0),
          "secs_since_last_txn": (0.0, 2_592_000.0, 60.0), "distinct_merchants_24h": (0, 40, 1),
          "distinct_countries_7d": (0, 8, 1), "amount_to_avg_30d": (0.0, 20.0, 0.1), "amount_zscore_30d": (-5.0, 15.0, 0.1),
          "hour_deviation": (0.0, 12.0, 0.1)}
BINARY = {"is_new_merchant", "is_new_device", "is_foreign_country"}

st.set_page_config(page_title="Fraud detection demo", page_icon="🛡️", layout="wide")


@st.cache_resource
def local_model():
    d = ROOT / "artifacts"
    return (joblib.load(d / "model.joblib"), json.loads((d / "model_meta.json").read_text()),
            json.loads((d / "baseline.json").read_text()))


def typical_values(baseline: dict) -> dict:
    """A 'normal' value per feature from the validation baseline: the most common value (flags) or the median bin edge."""
    out = {}
    for name, spec in baseline["features"].items():
        if spec["type"] == "categorical":
            out[name] = spec["values"][int(np.argmax(spec["expected"][:-1]))]
        else:
            out[name] = spec["edges"][len(spec["edges"]) // 2]
    return out


def score_remote(values: dict) -> dict:
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    body = {"instances": [{"transaction_id": 1, **values}]}
    r = rt.invoke_endpoint(EndpointName=ENDPOINT, ContentType="application/json", Body=json.dumps(body),
                           CustomAttributes=f"demo-{uuid.uuid4().hex[:8]}")
    out = json.loads(r["Body"].read())
    return {**out["predictions"][0], "threshold": out["threshold"], "backend": "SageMaker endpoint"}


def score_local(values: dict) -> dict:
    model, meta, _ = local_model()
    p = float(model.predict_proba(np.array([[values[f] for f in feature_names()]], dtype=np.float32))[0, 1])
    return {"fraud_score": p, "confidence": max(p, 1 - p), "decision": "FRAUD" if p >= meta["threshold"] else "LEGIT",
            "threshold": meta["threshold"], "backend": "local model"}


def explain(values: dict) -> pd.DataFrame:
    """How much each input pushes the score: score now minus score with that one input set to a typical value."""
    model, _, baseline = local_model()
    typical = typical_values(baseline)
    names = feature_names()
    rows = [[values[f] for f in names]]
    for i, f in enumerate(names):
        r = list(rows[0])
        r[i] = typical[f]
        rows.append(r)
    p = model.predict_proba(np.array(rows, dtype=np.float32))[:, 1]
    df = pd.DataFrame({"Input": [short(f) for f in names], "Effect on fraud score": p[0] - p[1:]})
    return df.reindex(df["Effect on fraud score"].abs().sort_values(ascending=False).index).head(6)


def load_scenario():
    for f, v in SCENARIOS[st.session_state["scenario"]].items():
        if f in BINARY:
            st.session_state[f"in_{f}"] = bool(v)
        else:   # the widget's min/max/step types must match the stored value's type
            st.session_state[f"in_{f}"] = int(v) if isinstance(RANGES[f][2], int) else float(v)


# ------------------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛡️ Fraud detection")
    st.caption("Random Forest · DuckDB features · AWS SageMaker · CloudWatch")
    backend = st.radio("Score with", ["Live SageMaker endpoint", "Local model (no AWS)"],
                       help="The live option calls the deployed endpoint and writes an audit record, like production traffic.")
    st.link_button("Open CloudWatch dashboard", DASHBOARD_URL)
    st.info("Built and tested on **synthetic** data. Real-world numbers would differ.")

tab_try, tab_perf, tab_live, tab_report = st.tabs(["Try it", "How good is it", "Live monitoring", "Manager report"])

# ------------------------------------------------------------------------------------------------ try it
with tab_try:
    st.subheader("Score a transaction")
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = "Everyday purchase"
        load_scenario()
    st.selectbox("Start from a scenario", list(SCENARIOS), key="scenario", on_change=load_scenario)
    st.caption(SCENARIO_NOTE[st.session_state["scenario"]])

    values, cols = {}, st.columns(3)
    for i, f in enumerate(feature_names()):
        with cols[i % 3]:
            if f in BINARY:
                values[f] = int(st.checkbox(label(f), key=f"in_{f}", help=FEATURES[f][1]))
            else:
                lo, hi, step = RANGES[f]
                values[f] = st.number_input(label(f), lo, hi, step=step, key=f"in_{f}", help=FEATURES[f][1])

    if st.button("Score this transaction", type="primary"):
        try:
            res = score_remote(values) if backend.startswith("Live") else score_local(values)
        except Exception as exc:  # noqa: BLE001 - show the reason, then fall back so the demo never dead-ends
            st.warning(f"Could not reach the endpoint ({type(exc).__name__}). Using the local model instead.")
            res = score_local(values)
        flagged = res["decision"] == "FRAUD"
        left, right = st.columns([1, 2])
        with left:
            (st.error if flagged else st.success)(f"{'⚠️ FLAGGED AS FRAUD' if flagged else '✅ Looks legitimate'}")
            st.metric("Fraud score", f"{res['fraud_score']:.3f}", help=f"Flagged at {res['threshold']:.3f} or above. "
                      "A ranking score, not a probability.")
            st.caption(f"Scored by: {res['backend']}")
        with right:
            st.progress(min(res["fraud_score"], 1.0), text=f"score {res['fraud_score']:.3f}  ·  alert threshold {res['threshold']:.3f}")
            st.markdown("**What pushed the score** (explained with a local copy of the same model)")
            ex = explain(values)
            st.bar_chart(ex.set_index("Input"), horizontal=True, height=240)
            st.caption("Bars to the right raise the fraud score; to the left lower it. Each bar is the score change "
                       "when that one input is replaced by a typical value.")

# ------------------------------------------------------------------------------------------------ performance
with tab_perf:
    m = kpis.load_metrics()
    h = kpis.headline(m)
    st.subheader("Held-out test results")
    st.caption(f"{h['rows']:,} transactions from days {h['test_days'][0] + 1} to {h['test_days'][1] + 1}, never used in training "
               f"or to set the threshold. Model {h['model_version']}.")
    c = st.columns(4)
    c[0].metric("Fraud caught", f"{h['fraud_caught_pct']:.1f}%", help="Recall")
    c[1].metric("Alerts that are real", f"{h['alert_precision_pct']:.1f}%", help="Precision")
    c[2].metric("False alarms / 10k", f"{h['false_alarms_per_10k']:.1f}")
    c[3].metric("Fraud missed", f"{h['missed']:,}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Caught, by fraud type (%)**")
        st.bar_chart(pd.DataFrame({"Caught %": [t["recall_pct"] for t in kpis.by_type(m)]},
                                  index=[t["name"] for t in kpis.by_type(m)]), horizontal=True, height=220)
    with right:
        st.markdown("**What the model relies on**")
        imp = json.loads((ROOT / "reports" / "feature_importance.json").read_text())
        st.bar_chart(pd.DataFrame({"Influence": [100 * v for v in imp.values()]}, index=[short(k) for k in imp]),
                     horizontal=True, height=220)
    st.image(str(ROOT / "reports" / "pr_curve.png"), caption="Precision-recall curve on the test set", width=420)

# ------------------------------------------------------------------------------------------------ live
with tab_live:
    st.subheader("Live model health")
    hours = st.select_slider("Window (hours)", [1, 3, 6, 12, 24, 72], value=24)
    st.caption("This deployment only receives replayed test data and demo requests, so these figures show the monitoring "
               "working, not real customer activity. Audit-log metrics lag by a few minutes.")

    @st.cache_data(ttl=60, show_spinner="Reading CloudWatch…")
    def _live(hours: int):
        return live.health(hours), live.alarms(), live.psi_by_feature(hours)

    health, alarms, psi = _live(hours)
    if health is None:
        st.warning("CloudWatch could not be reached from here (no AWS credentials or no network).")
    else:
        c = st.columns(4)
        c[0].metric("Transactions scored", f"{health['predictions']:,}")
        c[1].metric("Flagged as fraud", "n/a" if health["flag_rate_pct"] is None else f"{health['flag_rate_pct']:.2f}%")
        c[2].metric("Avg confidence", "n/a" if health["confidence_avg"] is None else f"{health['confidence_avg']:.3f}")
        c[3].metric("Avg scoring time", "n/a" if health["latency_ms"] is None else f"{health['latency_ms']:.0f} ms")
        if alarms:
            icons = {"OK": "🟢", "ALARM": "🔴", "INSUFFICIENT_DATA": "⚪"}
            st.dataframe(pd.DataFrame([{"Check": a["label"], "Status": f"{icons.get(a['state'], '⚪')} {a['state'].title()}",
                                        "Since (UTC)": f"{a['changed']:%Y-%m-%d %H:%M}"} for a in alarms]),
                         hide_index=True, width="stretch")
        if psi:
            st.markdown("**Input drift by feature** (PSI; above 0.2 is a significant shift)")
            items = sorted(psi.items(), key=lambda kv: -kv[1])
            st.bar_chart(pd.DataFrame({"PSI": [v for _, v in items]},
                                      index=[short(k) for k, _ in items]),
                         horizontal=True, height=320)
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

# ------------------------------------------------------------------------------------------------ report
with tab_report:
    st.subheader("Generate a manager report")
    st.write("A one-page, plain-language summary: headline KPIs, what is caught and missed, estimated financial effect, "
             "live health and caveats. Opens in any browser and prints to PDF.")
    c1, c2, c3 = st.columns(3)
    r_hours = c1.select_slider("Live window (hours)", [1, 3, 6, 12, 24, 72], value=24, key="r_hours")
    r_cost = c2.number_input("Review cost per alert ($, assumption)", 0.0, 500.0, kpis.DEFAULT_REVIEW_COST, 0.5)
    r_live = c3.checkbox("Include live AWS health", value=True)
    if st.button("Generate report", type="primary"):
        with st.spinner("Building report…"):
            out = report.generate(Path(tempfile.gettempdir()) / "fraud_manager_report.html", r_hours, not r_live, r_cost)
            st.session_state["report_html"] = out.read_text(encoding="utf-8")
    if "report_html" in st.session_state:
        st.download_button("Download HTML report", st.session_state["report_html"], "fraud_manager_report.html", "text/html")
        components.html(st.session_state["report_html"], height=900, scrolling=True)
