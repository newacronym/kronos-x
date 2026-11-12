"""E0: does Kronos run, how fast is it, and does its output look sane?

Run on Modal from the project root:  uv run modal run modal_app.py --exp e0_smoke
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from kranos import RESULTS_DIR, VAL_END, VAL_START
from kranos.data import fetch_daily, quality_report
from kranos.kronos_api import best_device, load_predictor, n_params, predict_chunked

TICKERS = ["RELIANCE.NS"]
TICKER = TICKERS[0]
LOOKBACK, HORIZON = 90, 10  # the paper's daily backtest setting
N_ORIGINS = 100
MAX_SEQS = 1000  # series x samples per forward pass; halved automatically on out-of-memory
OUT = RESULTS_DIR / "e0_smoke"


def prepare():
    """Runs locally (CPU only) before the GPU job: download and cache the data it needs."""
    for t in TICKERS:
        fetch_daily(t)


def windows(df: pd.DataFrame, origins: list[int]):
    """For each origin index t: history rows [t-LOOKBACK+1, t] and the next HORIZON rows."""
    xs, xts, yts, futures = [], [], [], []
    for t in origins:
        hist = df.iloc[t - LOOKBACK + 1 : t + 1]
        fut = df.iloc[t + 1 : t + 1 + HORIZON]
        xs.append(hist[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
        xts.append(hist["timestamps"].reset_index(drop=True))
        yts.append(fut["timestamps"].reset_index(drop=True))
        futures.append(fut)
    return xs, xts, yts, futures


def run(predictor, xs, xts, yts, samples, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()
    preds, seqs = predict_chunked(predictor, xs, xts, yts, pred_len=HORIZON, sample_count=samples,
                                  max_seqs=MAX_SEQS, T=0.6, top_p=0.9)
    if predictor.device.startswith("cuda"):
        torch.cuda.synchronize()
    return preds, time.perf_counter() - t0, seqs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = best_device()
    report = {"ticker": TICKER, "lookback": LOOKBACK, "horizon": HORIZON, "torch": torch.__version__,
              "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None}
    print("device:", device, report["gpu"])

    df = fetch_daily(TICKER)
    report["data"] = quality_report(df)
    print("data:", report["data"])

    # Origins inside the validation window, each with a full future horizon available.
    in_val = df.index[(df["timestamps"] >= VAL_START) & (df["timestamps"] <= VAL_END)]
    origins = [t for t in in_val if t >= LOOKBACK - 1 and t + HORIZON < len(df)][:N_ORIGINS]
    xs, xts, yts, futures = windows(df, origins)
    report["origins"] = {"n": len(origins), "first": str(df["timestamps"][origins[0]].date()),
                         "last": str(df["timestamps"][origins[-1]].date())}

    # 1. Speed: one batch of 100 windows is roughly one trading day of a Nifty 100 backtest.
    timings = []
    for size in ["mini", "small", "base"]:
        pred = load_predictor(size, device)
        run(pred, xs[:4], xts[:4], yts[:4], 1)  # warm-up: CUDA init and kernel selection
        for samples in [1, 10]:
            _, secs, seqs = run(pred, xs, xts, yts, samples)
            row = {"model": size, "params_M": round(n_params(pred) / 1e6, 1), "series": len(xs),
                   "samples": samples, "seconds": round(secs, 2), "seqs_per_call": seqs}
            timings.append(row)
            print("timing:", row)
        del pred
        torch.cuda.empty_cache() if device.startswith("cuda") else None
    report["timings"] = timings

    # 2. Determinism and a first look at forecast quality (Kronos-small, N=10).
    pred = load_predictor("small", device)
    preds_a, _, _ = run(pred, xs, xts, yts, samples=10, seed=0)
    preds_b, _, _ = run(pred, xs, xts, yts, samples=10, seed=0)
    preds_c, _, _ = run(pred, xs, xts, yts, samples=10, seed=1)
    close = lambda ps: np.stack([p["close"].to_numpy() for p in ps])
    report["determinism"] = {
        "same_seed_max_abs_diff": float(np.abs(close(preds_a) - close(preds_b)).max()),
        "diff_seed_mean_abs_diff_pct": float(np.mean(np.abs(close(preds_a) / close(preds_c) - 1)) * 100),
    }
    print("determinism:", report["determinism"])

    last_close = np.array([x["close"].iloc[-1] for x in xs])
    actual_ret = np.array([f["close"].iloc[-1] for f in futures]) / last_close - 1
    pred_ret = close(preds_a)[:, -1] / last_close - 1
    momentum_ret = np.array([x["close"].iloc[-1] / x["close"].iloc[-21] - 1 for x in xs])  # 20-day momentum
    report["preview_single_stock"] = {
        "note": "100 overlapping 10-day windows on one stock; heavily autocorrelated, so a sanity check, not evidence",
        "direction_hit_rate_kronos": float(np.mean(np.sign(pred_ret) == np.sign(actual_ret))),
        "direction_hit_rate_always_up": float(np.mean(actual_ret > 0)),
        "rank_corr_kronos": float(spearmanr(pred_ret, actual_ret).statistic),
        "rank_corr_momentum20": float(spearmanr(momentum_ret, actual_ret).statistic),
        "mae_10d_return_kronos_pct": float(np.mean(np.abs(pred_ret - actual_ret)) * 100),
        "mae_10d_return_random_walk_pct": float(np.mean(np.abs(actual_ret)) * 100),
        "mean_abs_predicted_move_pct": float(np.mean(np.abs(pred_ret)) * 100),
        "mean_abs_actual_move_pct": float(np.mean(np.abs(actual_ret)) * 100),
    }
    print("preview:", report["preview_single_stock"])

    # 3. Picture: four origins spread across the validation window.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, i in zip(axes.flat, np.linspace(0, len(origins) - 1, 4).astype(int)):
        t = origins[i]
        hist = df.iloc[t - 40 : t + 1]
        fut = futures[i]
        ax.plot(hist["timestamps"], hist["close"], color="#16181D", lw=1.5, label="history")
        ax.plot(fut["timestamps"], fut["close"], color="#16181D", lw=1.5, ls="--", label="actual")
        ax.plot(fut["timestamps"], preds_a[i]["close"].to_numpy(), color="#2E43C2", lw=2, label="Kronos-small (N=10)")
        ax.set_title(f"origin {df['timestamps'][t].date()}", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{TICKER}: 90-day lookback, 10-day forecast (validation window)")
    fig.tight_layout()
    fig.savefig(OUT / "forecasts.png", dpi=130)

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "report.json")
    return OUT


if __name__ == "__main__":
    main()
