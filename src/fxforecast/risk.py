"""Risk and liquidity measures read off Monte Carlo paths: triggers, hedge matching, guarantee loss."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import TD


def trigger_probabilities(p: dict, base: np.ndarray, stress: np.ndarray,
                          ups=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """P(end above L) by formula and Monte Carlo, and P(touch L at any time) for base and stress."""
    T = base.shape[1] / TD
    rows = []
    for up in ups:
        L = p["s0"] * (1 + up)
        z = (np.log(L / p["s0"]) - p["mu"] * T) / (p["sigma"] * np.sqrt(T))
        rows.append({"level": f"+{up:.0%} ({L:.2f})",
                     "ends above: formula": 1 - norm.cdf(z),
                     "ends above: Monte Carlo": (base[:, -1] > L).mean(),
                     "touches at any time": (base.max(axis=1) >= L).mean(),
                     "touches, stress scenario": (stress.max(axis=1) >= L).mean()})
    return pd.DataFrame(rows).set_index("level")


def hedge_completion(paths: np.ndarray, notional: float, daily: float, trigger: float) -> dict:
    """Remitters sell `daily` USD on each day the rate is at or below `trigger`."""
    good_days = paths <= trigger
    cum = np.cumsum(good_days * daily, axis=1)
    filled = cum[:, -1]
    shortfall = np.clip(notional - filled, 0, None)
    reached = cum >= notional
    first_day = reached.argmax(axis=1) + 1
    days_to_fill = np.where(reached.any(axis=1), first_day, np.nan)
    return {"chance hedge completes": float((filled >= notional).mean()),
            "average shortfall (USD)": float(shortfall.mean()),
            "95% worst shortfall (USD)": float(np.percentile(shortfall, 95)),
            "median trading days to fill": float(np.nanmedian(days_to_fill)) if reached.any() else np.nan,
            "share of days with matching": float(good_days.mean())}


def guarantee_loss(paths: np.ndarray, notional: float, hedge_rate: float) -> dict:
    """Guarantor pays max(C/K - C/S_T, 0) with C = notional * K leones."""
    cash = notional * hedge_rate
    loss = np.clip(cash / hedge_rate - cash / paths[:, -1], 0, None)
    return {"chance guarantor pays": float((loss > 0).mean()),
            "expected loss (USD)": float(loss.mean()),
            "95% worst loss (USD)": float(np.percentile(loss, 95)),
            "expected loss (% of notional)": float(loss.mean() / notional * 100)}


def liquidity_table(scenarios: dict, hedge_rate: float, notional: float, daily: float,
                    buffers=(0.00, 0.02, 0.05, 0.10)) -> pd.DataFrame:
    rows = []
    for b in buffers:
        trigger = hedge_rate * (1 + b)
        for name, paths in scenarios.items():
            rows.append({"trigger K*": f"K +{b:.0%} ({trigger:.2f})", "scenario": name,
                         **hedge_completion(paths, notional, daily, trigger)})
    return pd.DataFrame(rows).set_index(["trigger K*", "scenario"])


def guarantee_table(scenarios: dict, s0: float, notional: float, ups=(0.00, 0.05, 0.10)) -> pd.DataFrame:
    rows = []
    for up in ups:
        K = s0 * (1 + up)
        for name, paths in scenarios.items():
            rows.append({"hedge rate K": f"S0 +{up:.0%} ({K:.2f})", "scenario": name,
                         **guarantee_loss(paths, notional, K)})
    return pd.DataFrame(rows).set_index(["hedge rate K", "scenario"])
