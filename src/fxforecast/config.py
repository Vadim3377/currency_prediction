"""Global settings and inflation inputs. Edit INFLATION_LIVE when new figures are published."""

from pathlib import Path

import numpy as np
import pandas as pd

START = "2015-01-01"
DATA_END = "2026-11-06"  # last trading day in the frozen snapshot; all analysis stops here
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

N_PATHS = 10_000  # simulated futures per forecast
SEED = 7
TD = 252  # trading days per year
M = 21  # trading days per month

HEADERS = {"User-Agent": "Mozilla/5.0 (currency-notebook)"}

TEST_ORIGIN = pd.Timestamp("2025-10-31")
TEST_END = pd.Timestamp("2026-10-30")
BACKTEST_DATES = pd.date_range("2017-12-31", "2024-09-30", freq="QE")

# Stress scenario: one 15% devaluation every two years on average
STRESS_RATE = 0.5
STRESS_SIZE = float(np.log(1.15))

# Liquidity placeholders (replace with Concept Note figures)
NOTIONAL_USD = 1_000_000  # N: dollars investors need over the year
DAILY_REMIT_USD = 10_000  # V: dollars remitters offer per trading day

OTHER = {"GHS": "Ghanaian cedi", "NGN": "Nigerian naira", "KES": "Kenyan shilling"}

# (inflation %, source) known on the forecast date
INFLATION_TEST = {  # forecast made on 31 Oct 2025
    "SLE": (5.36, "Statistics Sierra Leone, Sep 2025 y/y"),
    "GHS": (9.4, "Ghana Statistical Service, Sep 2025 y/y"),
    "NGN": (18.02, "NBS Nigeria (rebased CPI), Sep 2025 y/y"),
    "KES": (4.6, "KNBS Kenya, Sep 2025 y/y"),
    "USD": (3.0, "US BLS CPI-U, Sep 2025 y/y"),
}
INFLATION_LIVE = {  # forecast made on 6 Nov 2026
    "SLE": (15.66, "Statistics Sierra Leone, Aug 2026 y/y"),
    "GHS": (5.5, "Ghana Statistical Service, Oct 2026 y/y"),
    "NGN": (14.63, "NBS Nigeria, Sep 2026 y/y"),
    "KES": (7.3, "KNBS Kenya, Oct 2026 y/y"),
    "USD": (3.7, "US BLS CPI-U, Sep 2026 y/y"),
}
# Calendar-year average inflation, % (World Bank FP.CPI.TOTL.ZG via FRED: FPCPITOTLZGSLE, FPCPITOTLZGUSA)
ANNUAL_INFLATION = pd.DataFrame(
    {
        "SLE": [10.89, 18.22, 16.03, 14.80, 13.45, 11.87, 27.21, 47.64, 28.63],
        "USD": [1.26, 2.13, 2.44, 1.81, 1.23, 4.70, 8.00, 4.12, 2.95],
    },
    index=range(2016, 2025),
)


def backtest_inflation(origin: pd.Timestamp, code: str) -> float:
    """Previous calendar-year average: fully published before any date in `origin`'s year."""
    return float(ANNUAL_INFLATION.loc[origin.year - 1, code])
