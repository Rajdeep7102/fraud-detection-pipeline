"""Drives the Streamlit demo like a visitor would: pick a scenario, press the button, read the verdict.
Runs on the bundled model with no AWS (the default 'Local model' option)."""
import pytest

from fraud.config import ROOT, model_dir

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not (model_dir() / "model.joblib").exists(), reason="no model available")


def _score(scenario: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app" / "demo.py"), default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    at.selectbox(key="scenario").select(scenario)
    at.run()
    [b for b in at.button if b.label == "Score this transaction"][0].click()
    at.run()
    assert not at.exception, at.exception
    return at


def test_everyday_purchase_is_not_flagged():
    at = _score("Everyday purchase")
    assert any("Looks legitimate" in s.value for s in at.success)


def test_card_testing_burst_is_flagged():
    at = _score("Card testing burst")
    assert any("FLAGGED AS FRAUD" in e.value for e in at.error)


def test_default_backend_needs_no_aws():
    at = AppTest.from_file(str(ROOT / "app" / "demo.py"), default_timeout=90)
    at.run()
    assert at.radio[0].value.startswith("Local model")
