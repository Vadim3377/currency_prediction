"""fxforecast: inflation-drift Monte Carlo forecasts for frontier-market exchange rates.

Model:  ln S_{t+dt} = ln S_t + mu*dt + sigma*sqrt(dt)*eps_t,   dt = 1/252
  mu    = ln(1 + pi_local) - ln(1 + pi_US)      (relative purchasing power parity)
  sigma = std of 21-day log changes over the last 3 years, annualised
"""

from .model import (
    calibrate,
    describe,
    formula_bands,
    future_dates,
    horizon_until,
    no_drift,
    percentile_table,
    simulate,
)

__all__ = [
    "calibrate",
    "describe",
    "formula_bands",
    "future_dates",
    "horizon_until",
    "no_drift",
    "percentile_table",
    "simulate",
]
__version__ = "0.1.0"
