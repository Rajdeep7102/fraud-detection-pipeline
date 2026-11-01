"""On-demand manager report: one self-contained HTML file (charts embedded, print-friendly -> "Save as PDF").

    python -m fraud.reporting.report                       # reports/manager_report.html, with live AWS health
    python -m fraud.reporting.report --offline             # skip AWS; test-set KPIs only
    python -m fraud.reporting.report --hours 6 --review-cost 8 --out reports/today.html
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
from html import escape as esc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fraud.config import ROOT, model_dir  # noqa: E402
from fraud.reporting import kpis, live  # noqa: E402
from fraud.reporting.glossary import FEATURES, label, short  # noqa: E402

INK, MUTED, ACCENT, GOOD, WARN, BAD = "#1f2933", "#6b7785", "#2f6fdf", "#2e8b57", "#d98e04", "#c0392b"

CSS = """
:root{--ink:#1f2933;--muted:#6b7785;--line:#e3e8ee;--card:#f6f8fb;--accent:#2f6fdf}
*{box-sizing:border-box} body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
margin:0;background:#fff} main{max-width:900px;margin:0 auto;padding:32px 20px 56px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:34px 0 10px;padding-top:10px;border-top:1px solid var(--line)}
.sub{color:var(--muted);font-size:13px} .lead{font-size:16px;margin:18px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .v{font-size:28px;font-weight:650;line-height:1.1} .card .l{font-size:13px;margin-top:4px}
.card .n{font-size:12px;color:var(--muted);margin-top:6px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top} th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums} img{max-width:100%;height:auto}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:600;color:#fff}
.OK{background:#2e8b57}.ALARM{background:#c0392b}.INSUFFICIENT_DATA{background:#8a94a0}
.note{background:#fff8e6;border:1px solid #f1dfae;border-radius:8px;padding:10px 14px;font-size:13.5px;margin:12px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media(max-width:700px){.grid2{grid-template-columns:1fr}}
ul{margin:6px 0 6px 20px;padding:0} li{margin:3px 0}
@media print{main{padding:0} h2{break-after:avoid} .card,table,img{break-inside:avoid}}
"""


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _barh(labels: list[str], values: list[float], title: str, fmt: str, colors=None, xmax=None, vline=None) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(labels) + 0.9))
    y = range(len(labels))
    ax.barh(list(y), values, color=colors or ACCENT, height=0.62)
    ax.set_yticks(list(y), labels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax or max(values) * 1.18)
    for i, v in enumerate(values):
        ax.text(v + (xmax or max(values)) * 0.012, i, fmt.format(v), va="center", fontsize=9.5, color=INK)
    if vline:
        ax.axvline(vline[0], color=BAD, lw=1, ls="--")
        ax.text(vline[0], -0.7, vline[1], color=BAD, fontsize=8.5, ha="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors=MUTED)
    ax.set_title(title, fontsize=11, loc="left", color=INK, pad=10)
    return _png(fig)


def _card(value: str, lab: str, note: str = "") -> str:
    return f'<div class="card"><div class="v">{value}</div><div class="l">{esc(lab)}</div><div class="n">{esc(note)}</div></div>'


def _money(x: float) -> str:
    return f"${x:,.0f}"


TRAFFIC_NOTE = ("This deployment is a demonstration: its traffic is replayed test data and demo requests, so the figures "
                "below show the monitoring working, not real customer activity.")


def build_html(metrics: dict, importance: dict, money: dict | None, health: dict | None, alarms: list,
               psi: dict, generated: dt.datetime, hours: int, meta: dict | None = None) -> str:
    h = kpis.headline(metrics)
    types = kpis.by_type(metrics)
    d0, d1 = h["test_days"][0] + 1, h["test_days"][1] + 1

    # ---- charts
    type_chart = _barh([t["name"] for t in types], [t["recall_pct"] for t in types],
                       "Share of each fraud type caught (%)", "{:.1f}%",
                       colors=[GOOD if t["recall_pct"] >= 80 else WARN if t["recall_pct"] >= 40 else BAD for t in types],
                       xmax=110)
    top = list(importance.items())[:8]
    imp_chart = _barh([short(k) for k, _ in top], [100 * v for _, v in top], "What the model relies on most (% of influence)",
                      "{:.1f}%")

    # ---- sections
    lead = (f"On {h['rows']:,} held-out transactions (days {d0} to {d1}, never seen in training) the model raised "
            f"<b>{h['alerts']:,} alerts</b>. <b>{h['real_per_100_alerts']:.0f} of every 100</b> were real fraud, and it caught "
            f"<b>{h['fraud_caught_pct']:.0f}%</b> of all fraud. It works very well on card-testing attacks, partially on account "
            f"takeovers, and almost never spots stealth fraud.")
    cards = "".join([
        _card(f"{h['fraud_caught_pct']:.1f}%", "of fraud caught (recall)",
              f"95% range {h['recall_ci'][0]:.1f} to {h['recall_ci'][1]:.1f}%"),
        _card(f"{h['alert_precision_pct']:.1f}%", "of alerts are real fraud (precision)",
              f"95% range {h['precision_ci'][0]:.1f} to {h['precision_ci'][1]:.1f}%"),
        _card(f"{h['false_alarms_per_10k']:.1f}", "false alarms per 10,000 transactions",
              f"{h['false_alarms']:,} genuine purchases flagged"),
        _card(f"{h['missed']:,}", "fraudulent transactions missed", f"of {h['fraud_rows']:,} fraud in the test period"),
    ])
    type_rows = "".join(
        f"<tr><td><b>{esc(t['name'])}</b><br><span class='sub'>{esc(t['blurb'])}</span></td>"
        f"<td class='num'>{t['share_of_fraud_pct']:.0f}%</td><td class='num'>{t['rows']:,}</td>"
        f"<td class='num'>{t['caught']:,}</td><td class='num'><b>{t['recall_pct']:.1f}%</b></td></tr>" for t in types)
    types_html = (f"<table><tr><th>Fraud type</th><th class='num'>Share of all fraud</th><th class='num'>Transactions</th>"
                  f"<th class='num'>Caught</th><th class='num'>Caught %</th></tr>{type_rows}</table>"
                  f"<img alt='Caught by fraud type' src='{type_chart}'>")
    travel = metrics["by_kind"]["travel"]
    types_html += (f"<p class='sub'>Genuine trips abroad are the main source of false alarms: {travel['flagged']:,} of "
                   f"{travel['rows']:,} travel purchases ({100 * travel['flag_rate']:.1f}%) were flagged, against "
                   f"{100 * metrics['by_kind']['normal']['flag_rate']:.2f}% of ordinary purchases.</p>")

    if money:
        money_html = f"""
        <table>
          <tr><th>Item</th><th class='num'>Amount</th></tr>
          <tr><td>Total value of fraud in the test period</td><td class='num'>{_money(money['fraud_total'])}</td></tr>
          <tr><td>Value of fraud caught</td><td class='num'>{_money(money['fraud_caught'])} ({money['share_caught_pct']:.0f}%)</td></tr>
          <tr><td>Value of fraud missed</td><td class='num'>{_money(money['fraud_missed'])}</td></tr>
          <tr><td>Review cost: {money['alerts']:,} alerts x {_money(money['review_cost_per_alert'])} per alert <span class='sub'>(assumption)</span></td>
              <td class='num'>{_money(money['review_cost'])}</td></tr>
          <tr><td><b>Net benefit (fraud caught minus review cost)</b></td><td class='num'><b>{_money(money['net_benefit'])}</b></td></tr>
        </table>
        <div class='note'>Transaction amounts come from the synthetic dataset, so these dollars are illustrative of how the
        calculation works, not a forecast for a real portfolio. The review cost per alert is an assumption: change it with
        <code>--review-cost</code>. It also assumes a caught fraud is fully prevented.</div>"""
    else:
        money_html = "<p class='sub'>Dollar figures unavailable: run the evaluation step to produce the flagged-rows file.</p>"

    imp_rows = "".join(f"<tr><td><b>{esc(label(k))}</b></td><td>{esc(FEATURES[k][1])}</td></tr>" for k, _ in top[:5])
    drivers_html = (f"<img alt='Most influential inputs' src='{imp_chart}'>"
                    f"<table><tr><th>Input</th><th>Why it matters</th></tr>{imp_rows}</table>")

    if health is not None:
        def f(x, spec): return "n/a" if x is None else format(x, spec)
        live_cards = "".join([
            _card(f"{health['predictions']:,}", f"transactions scored (last {hours}h)"),
            _card(f(health["flag_rate_pct"], ".2f") + ("%" if health["flag_rate_pct"] is not None else ""), "flagged as fraud",
                  f"{health['flagged']:,} alerts; expected about {100 * h['alerts'] / h['rows']:.2f}% on normal traffic"),
            _card(f(health["confidence_avg"], ".3f"), "average model confidence", f"weakest 10%: {f(health['confidence_p10'], '.3f')}"),
            _card(f(health["latency_ms"], ".0f") + (" ms" if health["latency_ms"] is not None else ""), "average scoring time",
                  f"{health['errors_5xx']} server errors, {health['rejected']} rejected requests"),
        ])
        alarm_rows = "".join(f"<tr><td>{esc(a['label'])}</td><td><span class='pill {a['state']}'>{a['state'].replace('_', ' ')}</span></td>"
                             f"<td class='sub'>since {a['changed']:%Y-%m-%d %H:%M} UTC</td></tr>" for a in alarms)
        alarm_html = f"<table><tr><th>Check</th><th>Status</th><th></th></tr>{alarm_rows}</table>" if alarms else ""
        psi_html = ""
        if psi:
            items = sorted(psi.items(), key=lambda kv: -kv[1])[:8]
            psi_chart = _barh([short(k) for k, _ in items], [v for _, v in items],
                              "Input drift by feature (PSI; above 0.2 = significant)", "{:.2f}",
                              colors=[BAD if v > 0.2 else WARN if v > 0.1 else GOOD for _, v in items], vline=(0.2, "0.2"))
            psi_html = (f"<p class='sub'>Drift compares live inputs with what the model saw in validation. The latest drift "
                        f"check is shown (most recent {hours}h).</p><img alt='Drift by feature' src='{psi_chart}'>")
        live_html = f"<div class='note'>{esc(TRAFFIC_NOTE)}</div><div class='cards'>{live_cards}</div>{alarm_html}{psi_html}"
    else:
        live_html = ("<div class='note'>Live health was not included: AWS could not be reached from where this report was "
                     "generated (or it was run with <code>--offline</code>).</div>")

    if meta:
        v_rate = 100 * meta["valid_fraud_rows"] / meta["valid_rows"]
        prevalence = (f" Fraud was rarer in the test period ({100 * h['fraud_rows'] / h['rows']:.2f}% of transactions) than in the "
                      f"period used to set the alert threshold ({v_rate:.2f}%); with rarer fraud, the same rule produces a "
                      f"larger share of false alerts.")
    else:
        prevalence = " With rarer fraud, the same alert rule produces a larger share of false alerts."
    limits = f"""<ul>
      <li><b>The data is synthetic.</b> Real fraud is messier; these numbers show the pipeline works, not what a bank would see.</li>
      <li><b>Stealth fraud is essentially invisible</b> to these 12 inputs. Catching it needs richer signals (device and network data, customer history).</li>
      <li><b>Account takeovers are caught about half the time.</b> The first transaction of an episode looks normal; later ones are easier.</li>
      <li><b>The fraud score is a ranking, not a probability.</b> The model trained on a down-sampled set of genuine transactions.</li>
      <li><b>Precision depends on how common fraud is.</b>{prevalence}</li>
      <li><b>Next steps:</b> add device and customer-history signals, retrain on a schedule, and review the alert threshold with the fraud operations team.</li></ul>"""

    glossary = """<ul>
      <li><b>Recall (fraud caught):</b> of all real fraud, the share the model flagged.</li>
      <li><b>Precision (alerts that are real):</b> of everything flagged, the share that was real fraud.</li>
      <li><b>False alarm:</b> a genuine transaction flagged as fraud, which creates review work and customer friction.</li>
      <li><b>Drift (PSI):</b> how far live data has moved from the data the model learned on. Under 0.1 stable, over 0.2 significant.</li></ul>"""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fraud detection: manager report</title><style>{CSS}</style></head><body><main>
<h1>Fraud detection: manager report</h1>
<div class="sub">Generated {generated:%Y-%m-%d %H:%M} UTC &middot; model {esc(h['model_version'])} &middot; trained and tested on synthetic data</div>
<p class="lead">{lead}</p>
<div class="cards">{cards}</div>
<h2>What it catches</h2>{types_html}
<h2>Estimated financial effect</h2>{money_html}
<h2>What drives the decisions</h2>{drivers_html}
<h2>Live model health (last {hours} hours)</h2>{live_html}
<h2>Limits and caveats</h2>{limits}
<h2>Glossary</h2>{glossary}
</main></body></html>"""


def generate(out: Path | None = None, hours: int = 24, offline: bool = False,
             review_cost: float = kpis.DEFAULT_REVIEW_COST, now: dt.datetime | None = None) -> Path:
    metrics = kpis.load_metrics()
    importance = json.loads((ROOT / "reports" / "feature_importance.json").read_text())
    health = None if offline else live.health(hours)
    alarms = [] if offline or health is None else live.alarms()
    psi = {} if offline or health is None else live.psi_by_feature(hours)
    meta_path = model_dir() / "model_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    html = build_html(metrics, importance, kpis.dollars(review_cost), health, alarms, psi,
                      now or dt.datetime.now(dt.timezone.utc), hours, meta)
    out = out or ROOT / "reports" / "manager_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--review-cost", type=float, default=kpis.DEFAULT_REVIEW_COST)
    a = ap.parse_args()
    path = generate(a.out, a.hours, a.offline, a.review_cost)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
