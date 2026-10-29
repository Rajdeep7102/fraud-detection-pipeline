import numpy as np
import pytest

from fraud.monitoring import drift

rng = np.random.default_rng(0)


def test_same_distribution_is_stable():
    base = rng.lognormal(3, 1, 50_000)
    spec = drift.describe(base)
    assert spec["type"] == "quantile"
    assert drift.feature_psi(spec, rng.lognormal(3, 1, 20_000)) < 0.02


def test_shifted_distribution_is_flagged():
    spec = drift.describe(rng.lognormal(3, 1, 50_000))
    assert drift.feature_psi(spec, rng.lognormal(3.8, 1, 20_000)) > 0.2


def test_binary_feature_drift_is_detected():
    """The bug this guards against: quantile bins put every 0/1 value in one bin, so PSI was always 0."""
    base = (rng.random(50_000) < 0.01).astype(float)          # ~1% foreign
    spec = drift.describe(base)
    assert spec["type"] == "categorical" and spec["values"] == [0.0, 1.0]
    assert drift.feature_psi(spec, (rng.random(5_000) < 0.01).astype(float)) < 0.05
    assert drift.feature_psi(spec, (rng.random(5_000) < 0.20).astype(float)) > 0.2


def test_unseen_category_goes_to_other_bin():
    spec = drift.describe(np.array([0.0, 1.0] * 500))
    shares = drift.bin_shares(spec, np.array([0.0, 1.0, 7.0, 7.0]))
    assert shares.tolist() == [0.25, 0.25, 0.5]


def test_psi_of_identical_shares_is_zero():
    assert drift.psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)
