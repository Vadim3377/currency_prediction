"""Forecast scoring: average % error, 90% coverage, range width and CRPS; out-of-sample test and backtest."""

import numpy as np
import pandas as pd

from .config import BACKTEST_DATES, N_PATHS, TD, backtest_inflation
from .model import calibrate, no_drift, simulate

SCORE_COLS = ["average % error", "coverage of 90% range (%)",
              "12m range width (% of rate)", "CRPS at 12m (% of rate)"]
SHOCKS = ["normal", "t", "bootstrap"]


def crps(sample, y: float) -> float:
    """CRPS = E|X - y| - 0.5 E|X - X'|, computed exactly from a sorted sample."""
    x = np.sort(np.asarray(sample, dtype=np.float64))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / (n * n))


def score(paths: np.ndarray, actual: np.ndarray) -> dict:
    lo, mid, hi = np.percentile(paths, [5, 50, 95], axis=0)
    return {
        "average % error": float(np.mean(np.abs(mid - actual) / actual) * 100),
        "coverage of 90% range (%)": float(np.mean((actual >= lo) & (actual <= hi)) * 100),
        "12m range width (% of rate)": float((hi[-1] - lo[-1]) / mid[-1] * 100),
        "CRPS at 12m (% of rate)": float(crps(paths[:, -1], actual[-1]) / actual[-1] * 100),
    }


def competitors(p: dict, days: int, n: int = N_PATHS) -> dict:
    """Inflation model vs no-drift vs no-change benchmark."""
    return {
        "Inflation model": simulate(p, days, n),
        "No drift": simulate(no_drift(p), days, n),
        "No change": np.full((1, days), p["s0"], dtype=np.float32),
    }


def out_of_sample(prices: pd.Series, p: dict, end) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Score all competitors fitted at p['origin'] against the actual path up to `end`."""
    actual_s = prices[(prices.index > p["origin"]) & (prices.index <= end)]
    actual = actual_s.to_numpy()
    paths = competitors(p, len(actual))
    rows = []
    for name, pth in paths.items():
        lo, hi = np.percentile(pth[:, -1], [5, 95])
        rows.append({"model": name, **score(pth, actual),
                     "end: middle forecast": float(np.median(pth[:, -1])),
                     "end: 90% range": f"{lo:.2f} - {hi:.2f}"})
    return pd.DataFrame(rows).set_index("model"), actual_s, paths


def backtest(prices: pd.Series, dates=BACKTEST_DATES, code: str = "SLE", n: int = 2000) -> pd.DataFrame:
    """Repeat fit-forecast-score on each date (inflation = previous calendar-year average)."""
    rows = []
    for d in dates:
        p = calibrate(prices, d, backtest_inflation(d, code), backtest_inflation(d, "USD"))
        future = prices[prices.index > p["origin"]].iloc[:TD].to_numpy()
        for name, paths in competitors(p, len(future), n=n).items():
            rows.append({"start": p["origin"], "year": p["origin"].year, "model": name,
                         "drift used (%/yr)": p["mu"] * 100, **score(paths, future)})
    return pd.DataFrame(rows)


def compare_shocks(prices: pd.Series, p_test: dict, actual: np.ndarray,
                   dates=BACKTEST_DATES, code: str = "SLE", n: int = 2000) -> pd.DataFrame:
    """Normal vs t vs bootstrap shocks on the test year and averaged over the backtest dates."""
    test = pd.DataFrame([{"shocks": k, **score(simulate(p_test, len(actual), shocks=k), actual)}
                         for k in SHOCKS]).set_index("shocks")
    rows = []
    for d in dates:
        p = calibrate(prices, d, backtest_inflation(d, code), backtest_inflation(d, "USD"))
        future = prices[prices.index > p["origin"]].iloc[:TD].to_numpy()
        for k in SHOCKS:
            rows.append({"shocks": k, **score(simulate(p, len(future), n=n, shocks=k), future)})
    bt = pd.DataFrame(rows).groupby("shocks")[SCORE_COLS].mean().reindex(SHOCKS)
    return pd.concat({"test year": test, "backtest average": bt})


def shock_tails(p: dict, days: int = TD) -> pd.DataFrame:
    """1-day tails differ by shock type; 12-month percentiles barely do (central limit theorem)."""
    rows = {}
    for k in SHOCKS:
        paths = simulate(p, days, shocks=k)
        day1 = np.log(paths[:, 0] / p["s0"])
        end = paths[:, -1]
        rows[k] = {"1-day move: excess kurtosis": float(pd.Series(day1).kurt()),
                   "1-day move: 99.9% quantile (%)": float(np.percentile(day1, 99.9) * 100),
                   "12m: 1%": float(np.percentile(end, 1)),
                   "12m: middle": float(np.median(end)),
                   "12m: 99%": float(np.percentile(end, 99)),
                   "touches +10% within a year": float((paths.max(axis=1) >= p["s0"] * 1.1).mean())}
    return pd.DataFrame(rows).T
