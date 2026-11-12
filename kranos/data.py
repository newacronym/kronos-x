"""Daily OHLCV download from Yahoo Finance with a local parquet cache."""

import time

import pandas as pd
import yfinance as yf

from kranos import DATA_DIR

RAW_DIR = DATA_DIR / "raw" / "daily"
UNIVERSE_DIR = DATA_DIR / "universe"
# NSE's official list as downloaded on that date. Using today's members for a 2024-2026 test
# carries survivorship bias: stocks that rose into the index are over-represented.
NIFTY100_FILE = UNIVERSE_DIR / "nifty100_2026-11-12.csv"

# Price jumps Yahoo did not adjust for (demergers and similar), found in E1. A forecast window
# whose history or future crosses one of these dates is skipped for that stock.
CORPORATE_ACTION_BREAKS = {
    "MOTHERSON.NS": ["2022-01-14"],  # +28.6%, after the Samvardhana Motherson demerger
    "TMPV.NS": ["2025-10-14"],       # -40.2%, Tata Motors demerger into TMPV and TMCV
    "TRENT.NS": ["2026-01-01"],      # -33.0%, unadjusted corporate action
    "VEDL.NS": ["2026-04-30"],       # -64.9%, Vedanta demerger
}


def nifty100() -> pd.DataFrame:
    """Symbol, Yahoo ticker and NSE industry for each Nifty 100 member."""
    df = pd.read_csv(NIFTY100_FILE)
    return pd.DataFrame({"symbol": df["Symbol"], "ticker": df["Symbol"] + ".NS", "industry": df["Industry"]})


def load_cached(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Cached bars only (no network); tickers without a cache file are skipped."""
    return {t: pd.read_parquet(RAW_DIR / f"{t}.parquet") for t in tickers if (RAW_DIR / f"{t}.parquet").exists()}


def fetch_daily(ticker: str, start: str = "2015-01-01", refresh: bool = False) -> pd.DataFrame:
    """Split- and dividend-adjusted daily bars with columns timestamps, open, high, low, close, volume."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    for attempt in range(3):
        raw = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False, multi_level_index=False)
        if not raw.empty:
            break
        time.sleep(2 * (attempt + 1))  # Yahoo rate-limits bursts of requests
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = (
        raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        .rename_axis("timestamps")
        .reset_index()
    )
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    # Yahoo occasionally emits zero-volume holiday rows or NaN bars; Kronos rejects NaNs.
    df = df.dropna().query("volume > 0")
    # Today's bar is still forming while the market is open; never let a partial candle in.
    df = df[df["timestamps"] < pd.Timestamp.now().normalize()].reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    gaps = df["timestamps"].diff().dt.days
    jumps = (df["open"] / df["close"].shift() - 1).abs()
    return {
        "rows": len(df),
        "first": str(df["timestamps"].iloc[0].date()),
        "last": str(df["timestamps"].iloc[-1].date()),
        "max_calendar_gap_days": int(gaps.max()),
        "bars_with_gap_over_10pct": int((jumps > 0.10).sum()),
        "bad_ohlc_rows": int(((df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))).sum()),
    }
