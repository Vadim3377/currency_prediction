"""Download, parse, clean, splice and verify the daily exchange-rate data."""

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR, DATA_END, HEADERS, START


def download_yahoo_csv(symbol: str, start: str, path: Path) -> None:
    """Daily closes from the Yahoo chart API, written as date,close CSV."""
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp.now(tz="UTC").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"period1": p1, "period2": p2, "interval": "1d"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s").date,
        "close": res["indicators"]["quote"][0]["close"],
    })
    df.to_csv(path, index=False)


def download_wise_csv(source: str, target: str, path: Path) -> None:
    """Daily mid-market rates from Wise (max 5 years back), written as date,close CSV."""
    url = "https://wise.com/rates/history+live"
    params = {"source": source, "target": target, "length": 5,
              "resolution": "daily", "unit": "year"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = pd.DataFrame({
        "date": pd.to_datetime(df["time"], unit="ms").dt.date,
        "close": df["value"],
    })
    df.to_csv(path, index=False)


def read_rate_csv(path: Path, scale: float = 1.0) -> pd.Series:
    """One value per day (the live tick duplicates today), sorted, divided by `scale`."""
    df = pd.read_csv(path, parse_dates=["date"])
    s = df.dropna(subset=["close"]).groupby("date")["close"].last().sort_index()
    return s / scale


def despike(s: pd.Series, window: int = 11, tol: float = 0.05, verbose: bool = True) -> pd.Series:
    """Drop points more than `tol` away from their centred rolling median (bad ticks)."""
    med = s.rolling(window, center=True, min_periods=3).median()
    keep = (s / med - 1).abs() <= tol
    if verbose:
        print(f"  {s.name}: dropped {(~keep).sum()} spikes")
    return s[keep]


def to_trading_days(s: pd.Series, end: str = DATA_END) -> pd.Series:
    """Monday-Friday only, one value per day, cut at `end`."""
    s = s[s.index.dayofweek < 5]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s[:end]


# ---------------------------------------------------------------- snapshot manifest

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_info(path: Path) -> dict:
    df = pd.read_csv(path)
    return {"sha256": sha256(path), "rows": len(df),
            "first_date": str(df.iloc[0, 0]), "last_date": str(df.iloc[-1, 0])}


def verify_snapshot(files, manifest_path: Path, refresh: bool = False) -> dict:
    """Record files in the manifest (refresh / first sight) or stop if any file differs from it."""
    if manifest_path.exists() and not refresh:
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"created": date.today().isoformat(), "data_end": DATA_END, "files": {}}

    changed = False
    for p in files:
        known = manifest["files"].get(p.name)
        if known is None or refresh:
            manifest["files"][p.name] = file_info(p)
            changed = True
        elif sha256(p) != known["sha256"]:
            raise RuntimeError(f"{p.name} differs from the frozen snapshot. "
                               "Restore the original file or rerun with --refresh on purpose.")
    if changed:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------- loaders

def build_usd_sle(data_dir: Path = DATA_DIR, refresh: bool = False) -> pd.Series:
    """Download (if needed), clean and splice Yahoo + Wise into one USD/SLE daily series.

    Redenomination on 1 Jul 2022: 1 SLE = 1000 SLL, so all SLL quotes are divided by 1000.
    Returns trading-day rates up to DATA_END, in new leones per USD.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_yahoo = data_dir / "usd_sll_yahoo_raw.csv"
    raw_wise = data_dir / "usd_sll_wise_raw.csv"
    merged = data_dir / "usd_sle_daily.csv"

    if refresh or not raw_yahoo.exists():
        download_yahoo_csv("SLL=X", START, raw_yahoo)
    if refresh or not raw_wise.exists():
        download_wise_csv("USD", "SLL", raw_wise)

    yahoo = read_rate_csv(raw_yahoo, scale=1000)
    wise = read_rate_csv(raw_wise, scale=1000)
    yahoo.name, wise.name = "yahoo", "wise"
    yahoo_clean = despike(yahoo)
    wise_clean = despike(wise)

    # Splice: Yahoo up to the day before Wise starts, Wise from then on
    splice_date = wise_clean.index.min()
    rate = pd.concat([
        yahoo_clean[(yahoo_clean.index >= START) & (yahoo_clean.index < splice_date)]
        .to_frame("usd_sle").assign(source="yahoo"),
        wise_clean.to_frame("usd_sle").assign(source="wise"),
    ])
    rate.index.name = "date"
    rate["usd_sle"] = rate["usd_sle"].round(5)
    rate.to_csv(merged)
    print(f"Saved {merged}: {len(rate):,} rows, splice at {splice_date.date()}")

    verify_snapshot([raw_yahoo, raw_wise, merged], data_dir / "snapshot_manifest.json", refresh)

    s = pd.read_csv(merged, parse_dates=["date"], index_col="date")["usd_sle"]
    return to_trading_days(s)


def load_pair(target: str, data_dir: Path = DATA_DIR, refresh: bool = False) -> pd.Series:
    """Daily USD->target rates from Wise, cleaned and cut to trading days up to DATA_END."""
    raw = data_dir / f"usd_{target.lower()}_wise_raw.csv"
    if refresh or not raw.exists():
        download_wise_csv("USD", target, raw)
    verify_snapshot([raw], data_dir / "snapshot_manifest.json", refresh)
    px = read_rate_csv(raw)
    px.name = target
    return to_trading_days(despike(px))
