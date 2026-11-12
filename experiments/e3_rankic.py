"""E3: does Kronos rank Nifty 100 stocks by their next 10-day return better than simple factors?

Validation window only. GPU part on Modal:  uv run modal run modal_app.py --exp e3_rankic
Re-run the analysis locally from saved paths (CPU):  uv run python -m experiments.e3_rankic --analyze
"""

import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from kranos import RESULTS_DIR, VAL_END, VAL_START
from kranos.data import fetch_daily, load_cached, nifty100
from kranos.forecast import forecast_days

MODEL, N_PATHS, T, TOP_P, SEED = "small", 10, 0.6, 0.9, 0  # the paper's forecasting settings, untuned
LOOKBACK, HORIZON = 90, 10
MIN_STOCKS = 20  # skip a day's IC if fewer stocks than this have a clean window
OUT = RESULTS_DIR / "e3_rankic_val"


def prepare():
    from experiments.e1_data import prepare as e1_prepare
    e1_prepare()


def load_universe():
    uni = nifty100()
    bars = load_cached(list(uni["ticker"]))
    cal = fetch_daily("^NSEI")["timestamps"]
    return uni, bars, cal


def main():
    from kranos.kronos_api import best_device, load_predictor

    OUT.mkdir(parents=True, exist_ok=True)
    _, bars, cal = load_universe()
    days = cal[(cal >= VAL_START) & (cal <= VAL_END)].tolist()

    device = best_device()
    pred = load_predictor(MODEL, device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t0 = time.perf_counter()
    paths, index = forecast_days(pred, bars, days, lookback=LOOKBACK, horizon=HORIZON, n_paths=N_PATHS, T=T,
                                 top_p=TOP_P, last_target_day=VAL_END, min_stocks=MIN_STOCKS)
    np.savez_compressed(OUT / "paths.npz", paths=paths)
    index.to_parquet(OUT / "index.parquet", index=False)
    run = {"model": MODEL, "n_paths": N_PATHS, "T": T, "top_p": TOP_P, "seed": SEED, "lookback": LOOKBACK,
           "horizon": HORIZON, "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
           "gpu_seconds": round(time.perf_counter() - t0, 1)}
    (OUT / "run.json").write_text(json.dumps(run, indent=2))
    analyze()
    return OUT


# ---------------------------------------------------------------- analysis (CPU) ---

def factor_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Simple baseline signals and forward-return targets for every (date, ticker)."""
    frames = []
    for t, df in bars.items():
        c = df.set_index("timestamps")["close"]
        logret = np.log(c).diff()
        frames.append(pd.DataFrame({
            "ticker": t,
            "rev5": -(c / c.shift(5) - 1),          # short-term reversal: last week's losers
            "rev10": -(c / c.shift(10) - 1),
            "mom20": c / c.shift(20) - 1,           # one-month momentum
            "mom60_skip5": c.shift(5) / c.shift(60) - 1,
            "lowvol20": -logret.rolling(20).std(),  # low-volatility anomaly
            "fwd1": c.shift(-1) / c - 1,
            "fwd5": c.shift(-5) / c - 1,
            "fwd10": c.shift(-HORIZON) / c - 1,
            "close10": c.shift(-HORIZON),
        }).rename_axis("date").reset_index())
    return pd.concat(frames, ignore_index=True)


def daily_ic(df: pd.DataFrame, sig: str, target: str) -> pd.Series:
    def one(x):
        return x[sig].rank().corr(x[target].rank()) if len(x) >= MIN_STOCKS else np.nan
    return df[["date", sig, target]].dropna().groupby("date").apply(one, include_groups=False).dropna()


def summarize(ic: pd.Series, horizon: int) -> dict:
    n, m, s = len(ic), ic.mean(), ic.std()
    return {"mean_ic": round(m, 4), "std": round(s, 4), "icir": round(m / s, 3), "pct_days_positive": round((ic > 0).mean(), 3),
            "days": n, "independent_periods": round(n / horizon, 1),
            # t-stat with the sample size cut to non-overlapping periods; a rough guard against overlap
            "t_stat_overlap_adjusted": round(m / s * np.sqrt(n / horizon), 2)}


def residualize(df: pd.DataFrame, sig: str, controls: list[str]) -> pd.Series:
    """Per day: the part of sig's cross-sectional rank not explained by the controls' ranks."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.dropna(subset=[sig] + controls).groupby("date"):
        y = g[sig].rank().to_numpy()
        X = np.column_stack([np.ones(len(g))] + [g[c].rank().to_numpy() for c in controls])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[g.index] = y - X @ beta
    return out


def quintile_spread(df: pd.DataFrame, sig: str, target: str = "fwd10") -> float:
    def one(x):
        q = x[sig].rank(pct=True)
        return x.loc[q > 0.8, target].mean() - x.loc[q <= 0.2, target].mean()
    return float(df[["date", sig, target]].dropna().groupby("date").apply(one, include_groups=False).mean())


def analyze(out=OUT):
    idx = pd.read_parquet(out / "index.parquet")
    paths = np.load(out / "paths.npz")["paths"]
    run = json.loads((out / "run.json").read_text())
    uni, bars, _ = load_universe()

    r = paths[..., 3] / idx["close0"].to_numpy()[:, None, None] - 1  # (obs, paths, steps) forecast returns
    last = r[:, :, -1]
    df = idx.copy()
    df["kronos_mean"] = r.mean(axis=(1, 2))   # the paper's signal: mean forecast close over the horizon
    df["kronos_last"] = last.mean(1)          # mean forecast close at the horizon
    df["kronos_prob_up"] = (last > 0).mean(1)
    df["kronos_risk_adj"] = last.mean(1) / (last.std(1) + 1e-9)
    df["random"] = np.random.default_rng(0).standard_normal(len(df))
    df = df.merge(factor_panel(bars), on=["date", "ticker"], how="left")
    df = df.merge(uni[["ticker", "industry"]], on="ticker", how="left")

    kronos = ["kronos_mean", "kronos_last", "kronos_prob_up", "kronos_risk_adj"]
    baselines = ["rev5", "rev10", "mom20", "mom60_skip5", "lowvol20", "random"]
    controls = ["rev5", "rev10", "mom20", "lowvol20"]
    df["kronos_mean_resid"] = residualize(df, "kronos_mean", controls)
    df["kronos_mean_sector_neutral"] = df["kronos_mean"] - df.groupby(["date", "industry"])["kronos_mean"].transform("mean")

    report = {"run": run, "obs": len(df), "days": int(df["date"].nunique()),
              "stocks_per_day": {"min": int(df.groupby("date").size().min()), "max": int(df.groupby("date").size().max())},
              "first_day": str(df["date"].min().date()), "last_day": str(df["date"].max().date())}

    ics = {}
    for target, h in [("fwd10", HORIZON), ("fwd5", 5), ("fwd1", 1)]:
        report[f"rank_ic_{target}"] = {}
        for s in kronos + ["kronos_mean_resid", "kronos_mean_sector_neutral"] + baselines:
            ic = daily_ic(df, s, target)
            report[f"rank_ic_{target}"][s] = summarize(ic, h)
            if target == "fwd10":
                ics[s] = ic

    report["top_minus_bottom_quintile_fwd10_pct"] = {s: round(quintile_spread(df, s) * 100, 3) for s in kronos + baselines}
    report["mean_daily_rank_corr_with_kronos_mean"] = {
        s: round(float(daily_ic(df, "kronos_mean", s).mean()), 3) for s in baselines[:-1] + ["fwd10"]}

    # Calibration: if paths and reality are exchangeable, the actual close lands inside the range
    # of N paths with probability (N-1)/(N+1).
    n = paths.shape[1]
    lo, hi, actual = paths[:, :, -1, 3].min(1), paths[:, :, -1, 3].max(1), df["close10"].to_numpy()
    ok = ~np.isnan(actual)
    report["calibration_10d"] = {
        "coverage_of_path_range": round(float(((actual >= lo) & (actual <= hi))[ok].mean()), 3),
        "expected_if_calibrated": round((n - 1) / (n + 1), 3),
        "mean_predicted_10d_return_pct": round(float(last.mean()) * 100, 3),
        "mean_actual_10d_return_pct": round(float(np.nanmean(df["fwd10"])) * 100, 3),
        "mean_abs_predicted_10d_return_pct": round(float(np.abs(last.mean(1)).mean()) * 100, 3),
        "mean_abs_actual_10d_return_pct": round(float(np.nanmean(np.abs(df["fwd10"]))) * 100, 3),
        "path_spread_std_pct": round(float(last.std(1).mean()) * 100, 3),
        "actual_cross_section_std_pct": round(float(df.groupby("date")["fwd10"].std().mean()) * 100, 3),
    }

    df.to_parquet(out / "signals.parquet", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    styles = {"kronos_mean": ("#2E43C2", 2.4), "kronos_mean_resid": ("#7D8BE0", 1.6), "rev5": ("#A86A10", 1.6),
              "mom20": ("#5A6070", 1.4), "lowvol20": ("#2F8F6B", 1.4), "random": ("#B5473A", 1.0)}
    for s, (col, lw) in styles.items():
        axes[0].plot(ics[s].index, ics[s].cumsum(), color=col, lw=lw, label=s)
    axes[0].axhline(0, color="#999", lw=0.8)
    axes[0].set_title("Cumulative daily RankIC vs. 10-day forward return (validation)", fontsize=10)
    axes[0].legend(fontsize=8)
    names = kronos + ["kronos_mean_resid", "kronos_mean_sector_neutral"] + baselines
    vals = [report["rank_ic_fwd10"][s]["mean_ic"] for s in names]
    axes[1].barh(names[::-1], vals[::-1], color=["#2E43C2" if s.startswith("kronos") else "#9CA2AE" for s in names[::-1]])
    axes[1].axvline(0, color="#999", lw=0.8)
    axes[1].set_title("Mean daily RankIC (fwd 10d)", fontsize=10)
    axes[1].tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "rankic.png", dpi=130)

    print(json.dumps({k: report[k] for k in ["days", "rank_ic_fwd10", "top_minus_bottom_quintile_fwd10_pct",
                                             "mean_daily_rank_corr_with_kronos_mean", "calibration_10d"]}, indent=2, default=str))
    return report


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else main()
