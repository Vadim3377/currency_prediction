"""Calibration, Monte Carlo simulation and closed-form bands of the inflation-drift model.

    ln S_{t+dt} = ln S_t + mu*dt + sigma*sqrt(dt)*eps_t

Shocks eps_t can be normal (default), Student t (fat tails, nu from kurtosis) or a
21-day block bootstrap of real past daily moves.
"""

import numpy as np
import pandas as pd

from .config import M, N_PATHS, SEED, TD


def calibrate(prices: pd.Series, origin, infl_local: float, infl_us: float, vol_years: int = 3) -> dict:
    """The model inputs, using data on or before `origin` only.

    infl_local, infl_us: annual inflation in %, as published before `origin`.
    """
    past = prices[:origin]
    recent = past[past.index[-1] - pd.DateOffset(years=vol_years):]
    lp = np.log(recent)
    monthly = (lp.shift(-M) - lp).dropna()
    daily = lp.diff().dropna()
    kurt = float(daily.kurt())  # excess kurtosis: 0 for a normal distribution
    nu = 30.0  # thin tails -> t is practically normal
    if kurt > 0:
        nu = min(max(4.0 + 6.0 / kurt, 4.1), 30.0)
    return {
        "origin": past.index[-1],
        "s0": float(past.iloc[-1]),
        "mu": float(np.log(1 + infl_local / 100) - np.log(1 + infl_us / 100)),
        "sigma": float(monthly.std() * np.sqrt(TD / M)),
        "nu": nu,  # t-distribution degrees of freedom
        "daily": (daily - daily.mean()).to_numpy(dtype=np.float32),  # past daily moves for the bootstrap
    }


def no_drift(p: dict) -> dict:
    """Same model with the inflation trend switched off (benchmark)."""
    q = dict(p)
    q["mu"] = 0.0
    return q


def everyday_steps(p: dict, rng: np.random.Generator, n: int, days: int, shocks: str) -> np.ndarray:
    """The random part of each daily log step, shape (n, days)."""
    dt = 1 / TD
    if shocks == "normal":
        return p["sigma"] * np.sqrt(dt) * rng.standard_normal((n, days), dtype=np.float32)
    if shocks == "t":
        nu = p["nu"]
        z = rng.standard_t(nu, size=(n, days)).astype(np.float32) * np.float32(np.sqrt((nu - 2) / nu))
        return p["sigma"] * np.sqrt(dt) * z
    if shocks == "bootstrap":
        hist = p["daily"]
        n_blocks = int(np.ceil(days / M))
        starts = rng.integers(0, len(hist) - M + 1, size=(n, n_blocks))
        idx = (starts[:, :, None] + np.arange(M)).reshape(n, -1)[:, :days]
        return hist[idx]
    raise ValueError(f"unknown shocks: {shocks}")


def simulate(p: dict, days: int, n: int = N_PATHS, seed: int = SEED,
             jump_rate: float = 0.0, jump_size: float = 0.0, shocks: str = "normal") -> np.ndarray:
    """Monte Carlo: n random daily paths of the model, shape (n, days).

    Optional stress jumps: `jump_rate` per year, each of log size `jump_size`. The everyday drift is
    reduced by jump_rate * jump_size so the middle forecast stays at the inflation trend.
    shocks: "normal", "t" (Student t, fat tails) or "bootstrap" (21-day blocks of real past moves).
    """
    rng = np.random.default_rng(seed)
    dt = 1 / TD
    mu = p["mu"] - jump_rate * jump_size
    steps = mu * dt + everyday_steps(p, rng, n, days, shocks)
    if jump_rate > 0:
        steps = steps + (rng.random((n, days), dtype=np.float32) < jump_rate * dt) * np.float32(jump_size)
    return (p["s0"] * np.exp(np.cumsum(steps, axis=1))).astype(np.float32)


def formula_bands(p: dict, days: int, z: float = 1.645):
    """Closed-form 5% / 50% / 95% for each future day (no simulation)."""
    T = np.arange(1, days + 1) / TD
    mid = p["s0"] * np.exp(p["mu"] * T)
    w = z * p["sigma"] * np.sqrt(T)
    return mid * np.exp(-w), mid, mid * np.exp(w)


def future_dates(origin, days: int) -> pd.DatetimeIndex:
    return pd.bdate_range(origin + pd.offsets.BDay(1), periods=days)


def horizon_until(origin, years: int) -> int:
    """Monday-Friday days from the day after `origin` to the same date `years` later."""
    return len(pd.bdate_range(origin + pd.offsets.BDay(1), origin + pd.DateOffset(years=years)))


def percentile_table(paths: np.ndarray, dates, qs=(5, 25, 50, 75, 95)) -> pd.DataFrame:
    pct = np.percentile(paths, qs, axis=0)
    return pd.DataFrame(pct.T, index=dates, columns=[f"p{q}" for q in qs])


def describe(p: dict) -> dict:
    return {
        "start rate S0": round(p["s0"], 3),
        "drift mu (% per yr)": round(p["mu"] * 100, 2),
        "volatility sigma (% per yr)": round(p["sigma"] * 100, 2),
        "t degrees of freedom nu": round(p["nu"], 1),
        "12m middle forecast": round(p["s0"] * np.exp(p["mu"]), 3),
        "12m 90% range": f"{p['s0'] * np.exp(p['mu'] - 1.645 * p['sigma']):.2f} - "
                         f"{p['s0'] * np.exp(p['mu'] + 1.645 * p['sigma']):.2f}",
    }
