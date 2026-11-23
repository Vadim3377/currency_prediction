"""Charts: history, fan charts of forecasts, multi-currency panels."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .evaluation import score
from .model import percentile_table

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
RED = "#e34948"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def fan(ax, dates, paths, history, actual_s=None) -> pd.DataFrame:
    pct = percentile_table(paths, dates)
    ax.plot(history.index, history.values, color=INK, linewidth=2, label="Actual (before forecast)")
    ax.fill_between(dates, pct["p5"], pct["p95"], color=RED, alpha=0.12, linewidth=0, label="90% range")
    ax.fill_between(dates, pct["p25"], pct["p75"], color=RED, alpha=0.25, linewidth=0, label="50% range")
    ax.plot(dates, pct["p50"], color=RED, linewidth=2.4, label="Middle forecast")
    if actual_s is not None:
        ax.plot(actual_s.index, actual_s.values, color=SERIES, linewidth=2, label="Actual (after forecast)")
    ax.axvline(history.index[-1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    style(ax)
    return pct


def plot_history(s: pd.Series, path):
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    ax.plot(s.index, s.values, color=SERIES, linewidth=2)
    redenom = pd.Timestamp("2022-07-01")
    ax.axvline(redenom, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(redenom - pd.Timedelta(days=20), s.max() * 0.8, "1 Jul 2022: redenomination\n1 SLE = 1000 SLL",
            color=INK2, fontsize=9, va="top", ha="right")
    first_d, first_v = s.index[0], s.iloc[0]
    last_d, last_v = s.index[-1], s.iloc[-1]
    for d, v, ha in [(first_d, first_v, "left"), (last_d, last_v, "right")]:
        ax.scatter([d], [v], s=40, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{v:.2f}\n{d:%d %b %Y}", (d, v), xytext=(0, 12), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=9, color=INK)
    growth = (last_v / first_v - 1) * 100
    ax.set_title(f"USD / SLE daily exchange rate, {first_d.year}-{last_d.year}", loc="left",
                 fontsize=15, color=INK, fontweight="bold", pad=24)
    ax.text(0, 1.015, f"New leones per 1 US dollar (pre-2022 SLL / 1000). "
                      f"The dollar costs {growth:,.0f}% more leones than in {first_d.year}.",
            transform=ax.transAxes, fontsize=10, color=INK2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_ylim(0, s.max() * 1.08)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_test(prices, p, actual_s, paths, path):
    o = p["origin"]
    fig, ax = plt.subplots(figsize=(12, 5.6), facecolor=SURFACE)
    fan(ax, actual_s.index, paths, prices[o - pd.DateOffset(years=2):o], actual_s)
    ax.set_title(f"USD / SLE: forecast made on {o:%d %b %Y} vs what happened", loc="left",
                 fontsize=14, fontweight="bold", color=INK)
    ax.set_ylabel("Leones per US dollar", color=INK2)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def plot_forecast(prices, p, dates, paths, path, history_years=2, title=None):
    o = p["origin"]
    fig, ax = plt.subplots(figsize=(12, 5.8), facecolor=SURFACE)
    pct = fan(ax, dates, paths, prices[o - pd.DateOffset(years=history_years):])
    ends = paths[:, -1]
    for q in (20, 50, 80):
        idx = int(np.argmin(np.abs(ends - np.percentile(ends, q))))
        ax.plot(dates, paths[idx], color=MUTED, linewidth=0.9, alpha=0.9,
                label="Example simulated futures" if q == 20 else None)
    for col, dy in (("p95", 2), ("p50", 2), ("p5", -10)):
        ax.annotate(f"{pct[col].iloc[-1]:.2f}", (dates[-1], pct[col].iloc[-1]), xytext=(6, dy),
                    textcoords="offset points", fontsize=10, color=RED)
    ax.set_title(title or f"USD / SLE forecast {o:%b %Y} - {dates[-1]:%b %Y}", loc="left",
                 fontsize=14, fontweight="bold", color=INK)
    ax.text(0, 1.01, f"Drift {p['mu']:.1%}/yr from the inflation gap, volatility {p['sigma']:.1%}/yr "
                     f"from monthly moves, {paths.shape[0]:,} Monte Carlo paths.",
            transform=ax.transAxes, fontsize=9.5, color=INK2)
    ax.set_ylabel("Leones per US dollar", color=INK2)
    if history_years > 2:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def plot_panels(series: dict, panels: dict, origin, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE)
    for ax, (code, (p, act, paths)) in zip(axes.ravel(), panels.items()):
        px = series[code]
        fan(ax, act.index, paths, px[p["origin"] - pd.DateOffset(years=2):p["origin"]], act)
        sc = score(paths, act.to_numpy())
        ax.set_title(f"USD / {code}: drift {p['mu']:.1%}/yr, coverage {sc['coverage of 90% range (%)']:.0f}%",
                     loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(f"Same model, several currencies: forecast made on {origin:%d %b %Y}", x=0.02, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
