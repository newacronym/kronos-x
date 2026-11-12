"""Cross-sectional forecasting loop shared by the ranking experiments (E3, E5, ...)."""

import time

import numpy as np
import pandas as pd

from kranos.data import CORPORATE_ACTION_BREAKS

X_COLS = ["open", "high", "low", "close", "volume"]


def build_windows(bars: dict[str, pd.DataFrame], day: pd.Timestamp, lookback: int, horizon: int, last_target_day: str):
    """Every stock with a clean `lookback`-bar history ending on `day` and `horizon` future bars.

    The future must end by `last_target_day`, so no price from a held-out window is used even as
    a target. Windows that cross a known unadjusted corporate action are skipped.
    """
    xs, xts, yts, meta = [], [], [], []
    for t, df in bars.items():
        pos = df.index[df["timestamps"] == day]
        if len(pos) == 0:
            continue
        i = pos[0]
        if i < lookback - 1 or i + horizon >= len(df):
            continue
        lo, hi = df["timestamps"].iloc[i - lookback + 1], df["timestamps"].iloc[i + horizon]
        if hi > pd.Timestamp(last_target_day):
            continue
        if any(lo <= pd.Timestamp(b) <= hi for b in CORPORATE_ACTION_BREAKS.get(t, [])):
            continue
        hist, fut = df.iloc[i - lookback + 1 : i + 1], df.iloc[i + 1 : i + 1 + horizon]
        xs.append(hist[X_COLS].reset_index(drop=True))
        xts.append(hist["timestamps"].reset_index(drop=True))
        yts.append(fut["timestamps"].reset_index(drop=True))
        meta.append({"date": day, "ticker": t, "close0": float(hist["close"].iloc[-1])})
    return xs, xts, yts, meta


def forecast_days(predictor, bars, days, *, lookback, horizon, n_paths, T, top_p, last_target_day, min_stocks=20, log_every=10):
    """Kronos paths for every clean (day, stock) window: (paths array, index DataFrame)."""
    from kranos.kronos_api import predict_paths

    metas, paths, t0 = [], [], time.perf_counter()
    for k, day in enumerate(days):
        xs, xts, yts, meta = build_windows(bars, day, lookback, horizon, last_target_day)
        if len(xs) < min_stocks:
            continue
        p, _ = predict_paths(predictor, xs, xts, yts, horizon, n_paths, T=T, top_p=top_p)
        paths.append(p)
        metas += meta
        if k % log_every == 0:
            print(f"day {k + 1}/{len(days)} {pd.Timestamp(day).date()} stocks={len(xs)} elapsed={time.perf_counter() - t0:.0f}s", flush=True)
    if not paths:
        return np.empty((0, n_paths, horizon, 6), dtype=np.float32), pd.DataFrame(columns=["date", "ticker", "close0"])
    return np.concatenate(paths), pd.DataFrame(metas)
