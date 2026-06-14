"""
Fetch the multi-asset ETF panel for the JSS 2026 application (excess returns).

Reads ``etf_universe.csv`` (ticker, sub_asset_class, asset_class, note) and
downloads **dividend- and split-adjusted** (total-return) prices via yfinance,
resamples to month-end, then converts to **excess** monthly log-returns by
subtracting the risk-free rate. This matches the MATF factor NAVs, which are
themselves computed on excess returns -- so the estimation Y (ETF excess
returns) and X (factor-NAV log-returns) sit on the same footing.

This is the public, reproducible counterpart to the production MATF universe:
anyone can re-run it from the shipped ticker list, with no Bloomberg terminal.

Risk-free rate
--------------
The 13-week US T-bill yield (Yahoo ``^IRX``, annualised percent) is converted to
a monthly log risk-free rate ``rf_log_t = log(1 + y_{t-1}/100) / 12`` and lagged
one month (the rate known at the start of month t funds month t's return). The
excess return is ``r_excess_t = log-return_t - rf_log_t``.

Why total-return (adjusted) prices
----------------------------------
For the bond, high-yield, EM and REIT sleeves the bulk of return is income, so
price-only series would badly distort estimated factor loadings. yfinance
``auto_adjust=True`` returns dividend+split-adjusted closes.

Outputs (OUT_DIR)
-----------------
  etf_prices.csv               month-end adjusted close, date x ticker
  etf_logreturns.csv           monthly total-return log returns, date x ticker
  etf_excess_logreturns.csv    monthly EXCESS log returns (estimation input)
  riskfree_monthly.csv         monthly log risk-free rate from ^IRX
  etf_meta.txt                 source, fetch date, coverage, rf summary

Requirements
------------
  pip install yfinance pandas
Network access to Yahoo Finance is required to run this (cannot run from a
sandbox restricted to package registries). A stooq fallback (pure urllib) is
documented in ``_stooq_fallback``; note stooq close is split-adjusted only.

Usage
-----
  python fetch_etf_panel.py                       # all tickers, from 2007-01
  python fetch_etf_panel.py --start 2013-01-01    # later balanced window
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "data"
UNIVERSE_CSV = Path(__file__).resolve().parent / "etf_universe.csv"
DEFAULT_START = "2017-01-01"
RF_TICKER = "^IRX"          # CBOE 13-week US T-bill yield, annualised percent


def _load_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(f"missing {UNIVERSE_CSV}")
    return pd.read_csv(UNIVERSE_CSV)


def _yf():
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover
        sys.exit("yfinance not installed -- `pip install yfinance` (or use the "
                 "stooq fallback documented in the module docstring).")
    return yf


def fetch_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Month-end total-return-adjusted close, date x ticker."""
    yf = _yf()
    raw = yf.download(tickers, start=start, auto_adjust=True,
                      progress=False, group_by="column")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.sort_index().resample("ME").last().reindex(columns=tickers)


def fetch_riskfree(start: str) -> pd.Series:
    """Monthly log risk-free rate from ^IRX, lagged one month."""
    yf = _yf()
    raw = yf.download(RF_TICKER, start=start, auto_adjust=False, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    yld = close.sort_index().resample("ME").last()       # annualised percent
    rf_log = np.log1p(yld / 100.0) / 12.0                 # monthly log rf
    rf_log = rf_log.shift(1)                              # rate set at start of month
    rf_log.name = "rf_log"
    return rf_log


def _stooq_fallback(ticker: str) -> pd.Series:  # pragma: no cover
    """Pure-urllib daily close from stooq (split-adjusted only)."""
    from urllib.request import urlopen
    import io
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    with urlopen(url, timeout=30) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Close"].rename(ticker)


def build(start: str) -> None:
    uni = _load_universe()
    tickers = uni["ticker"].tolist()
    print(f"Fetching {len(tickers)} ETFs from {start} (total-return adjusted) ...")
    prices = fetch_prices(tickers, start)
    rf_log = fetch_riskfree(start)

    logret = np.log(prices).diff().iloc[1:]
    rf_aligned = rf_log.reindex(logret.index)
    excess = logret.sub(rf_aligned, axis=0).dropna(how="all")

    first_valid = prices.apply(lambda s: s.first_valid_index())
    n_missing = prices.columns[prices.isna().all()].tolist()
    balanced_start = pd.to_datetime(first_valid.dropna()).max()
    cov = (uni.assign(first=first_valid.values)
              .groupby("sub_asset_class")["first"].max())
    rf_ann = float(np.expm1(rf_aligned.mean() * 12) * 100)   # sanity: ~3-5% recent decades

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUT_DIR / "etf_prices.csv")
    logret.to_csv(OUT_DIR / "etf_logreturns.csv")
    excess.to_csv(OUT_DIR / "etf_excess_logreturns.csv")
    rf_aligned.to_frame().to_csv(OUT_DIR / "riskfree_monthly.csv")
    with open(OUT_DIR / "etf_meta.txt", "w") as fh:
        fh.write("Multi-asset ETF panel for the JSS 2026 application (excess returns)\n")
        fh.write("Source: Yahoo Finance via yfinance (auto_adjust=True, total return)\n")
        fh.write(f"Risk-free: ^IRX 13-week T-bill, monthly log, lagged 1m; "
                 f"mean annualised ~{rf_ann:.2f}%\n")
        fh.write(f"Fetched: {pd.Timestamp.utcnow():%Y-%m-%d}\n")
        fh.write(f"Tickers: {len(tickers)}; missing: {n_missing or 'none'}\n")
        fh.write(f"Monthly obs: {len(excess)} "
                 f"({excess.index.min():%Y-%m}..{excess.index.max():%Y-%m})\n")
        fh.write(f"Balanced (all-present) start: {balanced_start:%Y-%m}\n\n")
        fh.write("Latest first-valid month by sub-asset class:\n")
        fh.write(cov.sort_values().to_string())

    print(f"  wrote etf_prices.csv             ({prices.shape[0]} x {prices.shape[1]})")
    print(f"  wrote etf_logreturns.csv         (total-return, {len(logret)} months)")
    print(f"  wrote etf_excess_logreturns.csv  (EXCESS, {len(excess)} months)  <- estimation input")
    print(f"  wrote riskfree_monthly.csv       (mean annualised rf ~{rf_ann:.2f}%)")
    print(f"  missing tickers: {n_missing or 'none'}")
    print(f"  balanced window starts: {balanced_start:%Y-%m}  "
          f"(unbalanced begins {excess.index.min():%Y-%m})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START, help="download start (YYYY-MM-DD)")
    build(ap.parse_args().start)


if __name__ == "__main__":
    main()
