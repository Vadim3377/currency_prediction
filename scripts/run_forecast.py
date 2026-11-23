"""End-to-end pipeline: data -> fit -> test -> backtest -> risk -> forecasts -> other currencies.

Usage (from the repository root):
    python scripts/run_forecast.py                 # frozen snapshot, all sections
    python scripts/run_forecast.py --skip-backtest # faster
    python scripts/run_forecast.py --refresh       # re-download data (breaks the snapshot)
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxforecast import config as C  # noqa: E402
from fxforecast import plotting  # noqa: E402
from fxforecast.data import build_usd_sle, load_pair  # noqa: E402
from fxforecast.evaluation import (SCORE_COLS, backtest, compare_shocks, competitors,  # noqa: E402
                                   out_of_sample, score, shock_tails)
from fxforecast.model import (calibrate, describe, formula_bands, future_dates,  # noqa: E402
                              horizon_until, percentile_table, simulate)
from fxforecast.risk import guarantee_table, liquidity_table, trigger_probabilities  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def header(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=C.OUTPUT_DIR)
    ap.add_argument("--refresh", action="store_true", help="re-download all data")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--skip-others", action="store_true", help="skip GHS / NGN / KES")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 1-5 data
    header("Data")
    s = build_usd_sle(args.data_dir, args.refresh)
    print(f"Trading days: {len(s):,} ({s.index.min().date()} -> {s.index.max().date()})")
    plotting.plot_history(s, out / "usd_sle_daily.png")

    # ------------------------------------------------------------ 8-9 fit + check
    header("Fit on data up to the test origin")
    p_test = calibrate(s, C.TEST_ORIGIN, C.INFLATION_TEST["SLE"][0], C.INFLATION_TEST["USD"][0])
    print(pd.Series(describe(p_test)).to_string())
    lo_f, mid_f, hi_f = formula_bands(p_test, C.TD)
    lo_s, mid_s, hi_s = np.percentile(simulate(p_test, C.TD), [5, 50, 95], axis=0)
    print("\nMonte Carlo vs formula at 12 months:")
    print(pd.DataFrame({"formula": [lo_f[-1], mid_f[-1], hi_f[-1]],
                        "Monte Carlo": [lo_s[-1], mid_s[-1], hi_s[-1]]},
                       index=["5%", "50%", "95%"]).round(3).to_string())

    # ------------------------------------------------------------ 10 out-of-sample
    header(f"Out-of-sample test {C.TEST_ORIGIN:%d %b %Y} -> {C.TEST_END:%d %b %Y}")
    table, actual_s, test_paths = out_of_sample(s, p_test, C.TEST_END)
    print(table.round(2).to_string())
    plotting.plot_test(s, p_test, actual_s, test_paths["Inflation model"], out / "test_usd_sle_2025_26.png")

    # ------------------------------------------------------------ 11 backtest
    if not args.skip_backtest:
        header(f"Backtest over {len(C.BACKTEST_DATES)} quarterly dates")
        bt = backtest(s)
        print(bt.groupby("model")[SCORE_COLS].mean().round(2).sort_values(SCORE_COLS[-1]).to_string())
        print("\nAverage % error by year:")
        print(bt.pivot_table(index="year", columns="model", values="average % error").round(1).to_string())
        bt.to_csv(out / "backtest.csv", index=False)

        header("Shock distribution: normal vs t vs bootstrap")
        print(f"nu estimated on {p_test['origin']:%d %b %Y}: {p_test['nu']:.1f}")
        print(compare_shocks(s, p_test, actual_s.to_numpy()).round(2).to_string())
        print()
        print(shock_tails(p_test).round(3).to_string())

    # ------------------------------------------------------------ 12-14 risk & liquidity
    header("Live fit, stress, triggers, liquidity")
    live_origin = s.index[-1]
    p_live = calibrate(s, live_origin, C.INFLATION_LIVE["SLE"][0], C.INFLATION_LIVE["USD"][0])
    live_days = horizon_until(live_origin, 1)
    live_dates = future_dates(live_origin, live_days)
    base = simulate(p_live, live_days)
    stress = simulate(p_live, live_days, jump_rate=C.STRESS_RATE, jump_size=C.STRESS_SIZE)
    scenarios = {"base": base, "stress": stress}

    print(f"Start {live_origin:%d %b %Y}, rate {p_live['s0']:.3f}; horizon ends {live_dates[-1]:%d %b %Y}\n")
    print(trigger_probabilities(p_live, base, stress).round(3).to_string())
    print(f"\nLiquidity: N = ${C.NOTIONAL_USD:,.0f}, V = ${C.DAILY_REMIT_USD:,.0f}/day, K = {p_live['s0']:.3f}")
    print(liquidity_table(scenarios, p_live["s0"], C.NOTIONAL_USD, C.DAILY_REMIT_USD).round(3).to_string())
    print()
    print(guarantee_table(scenarios, p_live["s0"], C.NOTIONAL_USD).round(3).to_string())

    # ------------------------------------------------------------ 15-16 forecasts
    header(f"12-month forecast from {live_origin:%d %b %Y}")
    print(pd.Series(describe(p_live)).to_string())
    live_pct = percentile_table(base, live_dates)
    live_pct.to_csv(out / "forecast_usd_sle_12m_percentiles.csv")
    np.save(out / "forecast_usd_sle_12m_paths.npy", base)
    print("\nMonthly checkpoints:")
    print(live_pct.iloc[C.M - 1::C.M][["p5", "p50", "p95"]].round(2).to_string())
    plotting.plot_forecast(s, p_live, live_dates, base, out / "forecast_usd_sle_12m.png")

    header("5-year scenario range")
    long_days = horizon_until(live_origin, 5)
    long_dates = future_dates(live_origin, long_days)
    long_paths = simulate(p_live, long_days, n=5000)
    long_pct = percentile_table(long_paths, long_dates)
    long_pct.to_csv(out / "forecast_usd_sle_5y_percentiles.csv")
    idx = [min(horizon_until(live_origin, y), long_days) - 1 for y in range(1, 6)]
    print(long_pct.iloc[idx][["p5", "p50", "p95"]].round(2).to_string())
    plotting.plot_forecast(s, p_live, long_dates, long_paths, out / "forecast_usd_sle_5y.png",
                           history_years=5, title="USD / SLE: 5 years of history and 5 years of simulated futures")

    # ------------------------------------------------------------ 17 other currencies
    if not args.skip_others:
        header("Other currencies")
        series = {"SLE": s}
        for code in C.OTHER:
            try:
                series[code] = load_pair(code, args.data_dir, args.refresh)
            except Exception as err:  # network / format problems should not kill the run
                print(f"  {code}: skipped ({err})")
        test_rows, live_rows, panels = [], [], {}
        for code, px in series.items():
            p = calibrate(px, C.TEST_ORIGIN, C.INFLATION_TEST[code][0], C.INFLATION_TEST["USD"][0])
            act = px[(px.index > p["origin"]) & (px.index <= C.TEST_END)]
            paths = competitors(p, len(act), n=5000)
            for name, pth in paths.items():
                test_rows.append({"currency": code, "model": name, **score(pth, act.to_numpy())})
            panels[code] = (p, act, paths["Inflation model"])
            q = calibrate(px, live_origin, C.INFLATION_LIVE[code][0], C.INFLATION_LIVE["USD"][0])
            live_rows.append({"currency": code, **describe(q)})
        print(pd.DataFrame(test_rows).set_index(["currency", "model"]).round(2).to_string())
        print("\nLive 12-month forecasts:")
        print(pd.DataFrame(live_rows).set_index("currency").to_string())
        plotting.plot_panels(series, panels, C.TEST_ORIGIN, out / "test_four_currencies.png")

    print(f"\nOutputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
