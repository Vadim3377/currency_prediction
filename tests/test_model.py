import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxforecast.evaluation import crps, score  # noqa: E402
from fxforecast.model import calibrate, formula_bands, simulate  # noqa: E402
from fxforecast.risk import guarantee_loss, hedge_completion  # noqa: E402


@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    steps = 0.10 / 252 + 0.15 / np.sqrt(252) * rng.standard_normal(len(idx))
    return pd.Series(20 * np.exp(np.cumsum(steps)), index=idx)


def test_calibrate_drift_is_inflation_gap(prices):
    p = calibrate(prices, pd.Timestamp("2024-06-30"), 15.0, 3.0)
    assert p["mu"] == pytest.approx(np.log(1.15) - np.log(1.03))
    assert p["origin"] <= pd.Timestamp("2024-06-30")
    assert 0.05 < p["sigma"] < 0.3


def test_monte_carlo_matches_formula(prices):
    p = calibrate(prices, pd.Timestamp("2024-06-30"), 15.0, 3.0)
    lo_f, mid_f, hi_f = formula_bands(p, 252)
    lo_s, mid_s, hi_s = np.percentile(simulate(p, 252, n=20_000), [5, 50, 95], axis=0)
    for a, b in [(lo_f[-1], lo_s[-1]), (mid_f[-1], mid_s[-1]), (hi_f[-1], hi_s[-1])]:
        assert b == pytest.approx(a, rel=0.01)


@pytest.mark.parametrize("shocks", ["normal", "t", "bootstrap"])
def test_simulate_shapes(prices, shocks):
    p = calibrate(prices, pd.Timestamp("2024-06-30"), 10.0, 3.0)
    paths = simulate(p, 100, n=500, shocks=shocks)
    assert paths.shape == (500, 100)
    assert np.all(paths > 0)


def test_stress_keeps_median(prices):
    p = calibrate(prices, pd.Timestamp("2024-06-30"), 10.0, 3.0)
    base = np.median(simulate(p, 252, n=20_000)[:, -1])
    stress = np.median(simulate(p, 252, n=20_000, jump_rate=0.5, jump_size=np.log(1.15))[:, -1])
    assert stress == pytest.approx(base, rel=0.05)


def test_crps_point_forecast_equals_abs_error():
    assert crps([5.0], 3.0) == pytest.approx(2.0)


def test_score_perfect_forecast():
    actual = np.linspace(10, 11, 50)
    sc = score(np.tile(actual, (3, 1)), actual)
    assert sc["average % error"] == pytest.approx(0)
    assert sc["coverage of 90% range (%)"] == 100


def test_hedge_and_guarantee():
    paths = np.array([[10.0] * 10, [12.0] * 10])
    h = hedge_completion(paths, notional=50, daily=10, trigger=11)
    assert h["chance hedge completes"] == 0.5
    g = guarantee_loss(paths, notional=100, hedge_rate=10)
    assert g["expected loss (USD)"] == pytest.approx((100 - 1000 / 12) / 2)
