import pytest

from fraud.reporting import kpis, live, report


@pytest.fixture(scope="module")
def metrics():
    return kpis.load_metrics()


def test_headline_kpis_are_consistent_with_the_confusion_matrix(metrics):
    h = kpis.headline(metrics)
    c = metrics["confusion"]
    assert h["caught"] + h["missed"] == h["fraud_rows"]
    assert h["alerts"] == c["tp"] + c["fp"]
    assert h["fraud_caught_pct"] == pytest.approx(100 * c["tp"] / (c["tp"] + c["fn"]), abs=0.01)
    assert h["alert_precision_pct"] == pytest.approx(100 * c["tp"] / (c["tp"] + c["fp"]), abs=0.01)
    assert h["false_alarms_per_10k"] == pytest.approx(10_000 * c["fp"] / h["rows"])
    assert h["real_per_100_alerts"] == pytest.approx(h["alert_precision_pct"], abs=0.01)


def test_by_type_covers_all_fraud_types(metrics):
    t = kpis.by_type(metrics)
    assert [x["key"] for x in t] == ["card_testing", "account_takeover", "stealth"]
    assert sum(x["share_of_fraud_pct"] for x in t) == pytest.approx(100, abs=0.01)


def test_offline_report_contains_the_measured_numbers_and_never_calls_aws(metrics, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("offline report must not call AWS")
    monkeypatch.setattr(live, "health", boom)
    monkeypatch.setattr(live, "alarms", boom)
    monkeypatch.setattr(live, "psi_by_feature", boom)

    out = report.generate(tmp_path / "r.html", offline=True)
    html = out.read_text(encoding="utf-8")
    h = kpis.headline(metrics)
    assert f"{h['fraud_caught_pct']:.1f}%" in html
    assert f"{h['alert_precision_pct']:.1f}%" in html
    assert f"{h['false_alarms_per_10k']:.1f}" in html
    assert "Live health was not included" in html
    assert "synthetic" in html.lower()                      # the caveat must always be present
    assert html.count("data:image/png;base64,") >= 2        # charts are embedded, so the file is self-contained
    assert "<script" not in html                            # a static document: nothing to execute


def test_dollars_use_dataset_amounts_and_add_up():
    d = kpis.dollars(review_cost=5.0)
    if d is None:
        pytest.skip("flagged-rows file or raw data not present")
    assert d["fraud_caught"] + d["fraud_missed"] == pytest.approx(d["fraud_total"])
    assert d["review_cost"] == pytest.approx(d["alerts"] * 5.0)
    assert d["net_benefit"] == pytest.approx(d["fraud_caught"] - d["review_cost"])
    assert 0 < d["share_caught_pct"] < 100


def test_live_helpers_degrade_gracefully_without_aws(monkeypatch):
    monkeypatch.setattr(live, "_session", lambda: (_ for _ in ()).throw(RuntimeError("no credentials")))
    assert live.health(1) is None and live.alarms() == [] and live.psi_by_feature(1) == {}
