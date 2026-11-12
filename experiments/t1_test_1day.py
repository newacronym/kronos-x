"""T1: the single pre-registered test-window run of E5's chosen signal (see findings.md, T1).

    uv run modal run modal_app.py --exp t1_test_1day     # GPU forecasts, then analysis
    uv run python -m experiments.t1_test_1day --analyze   # analysis only (CPU)
"""

import io
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import e3_rankic as e3
from kranos import RESULTS_DIR, TEST_START

CFG = {"model": "small", "lookback": 90, "horizon": 1, "n_paths": 10, "T": 0.6, "top_p": 0.9}
CONTROLS = ["rev5", "rev10", "mom20", "lowvol20"]
PASS_T = 2.0  # pre-registered
N_CHUNKS = 5
SEED = 9
OUT = RESULTS_DIR / "t1_test_1day"


def prepare():
    e3.prepare()


def orchestrate(forecast_chunk):
    _, _, cal = e3.load_universe()
    last_day = str(cal.iloc[-1].date())  # the last target day; origins stop one bar earlier
    days = [str(d.date()) for d in cal[cal >= TEST_START]]
    chunks = [[str(d) for d in c] for c in np.array_split(days, N_CHUNKS)]
    calls = [({**CFG, "last_target_day": last_day, "seed": SEED * 100 + i}, c) for i, c in enumerate(chunks)]
    print(f"launching {len(calls)} chunk jobs over {len(days)} test days, last target {last_day}")
    parts = list(forecast_chunk.starmap(calls, order_outputs=True))

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "paths.npy", np.concatenate([np.load(io.BytesIO(p["paths"])) for p in parts]))
    pd.concat([pd.read_parquet(io.BytesIO(p["index"])) for p in parts], ignore_index=True).to_parquet(OUT / "index.parquet", index=False)
    run = {**CFG, "last_target_day": last_day, "seed_base": SEED * 100, "chunks": N_CHUNKS, "gpu": parts[0]["gpu"],
           "gpu_seconds_total": round(sum(p["gpu_seconds"] for p in parts), 1)}
    (OUT / "run.json").write_text(json.dumps(run, indent=2))
    analyze()


def analyze():
    paths, index = np.load(OUT / "paths.npy"), pd.read_parquet(OUT / "index.parquet")
    run = json.loads((OUT / "run.json").read_text())
    _, bars, _ = e3.load_universe()

    df = index[["date", "ticker"]].copy()
    df["kronos_1d"] = (paths[:, :, 0, 3] / index["close0"].to_numpy()[:, None] - 1).mean(1)
    df = df.merge(e3.factor_panel(bars), on=["date", "ticker"], how="left")
    df["kronos_1d_resid"] = e3.residualize(df, "kronos_1d", CONTROLS)

    stripped_ic = e3.daily_ic(df, "kronos_1d_resid", "fwd1")
    raw_ic = e3.daily_ic(df, "kronos_1d", "fwd1")
    stripped = e3.summarize(stripped_ic, 1)
    verdict = "PASS" if stripped["mean_ic"] > 0 and stripped["t_stat_overlap_adjusted"] >= PASS_T else "FAIL"

    # Context only (not part of the verdict).
    def by_period(ic):
        g = ic.groupby(pd.PeriodIndex(ic.index, freq="Q").astype(str))
        return {k: {"mean_ic": round(v.mean(), 4), "days": len(v)} for k, v in g}

    def top_quintile_turnover(col):
        sets = df.dropna(subset=[col]).groupby("date").apply(
            lambda x: set(x.loc[x[col].rank(pct=True) > 0.8, "ticker"]), include_groups=False)
        prev = sets.shift()
        return float(np.mean([1 - len(a & b) / max(len(a), 1) for a, b in zip(sets[1:], prev[1:])]))

    report = {
        "run": run, "obs": len(df), "days": int(df["date"].nunique()),
        "first_day": str(df["date"].min().date()), "last_day": str(df["date"].max().date()),
        "stocks_per_day": {"min": int(df.groupby("date").size().min()), "max": int(df.groupby("date").size().max())},
        "verdict": {"criterion": f"factor-stripped RankIC > 0 and t >= {PASS_T}", "result": verdict},
        "factor_stripped_rank_ic_fwd1": stripped,
        "raw_rank_ic_fwd1": e3.summarize(raw_ic, 1),
        "baselines_rank_ic_fwd1": {s: e3.summarize(e3.daily_ic(df, s, "fwd1"), 1) for s in CONTROLS},
        "factor_stripped_by_quarter": by_period(stripped_ic),
        "rank_corr_with": {c: round(float(e3.daily_ic(df, "kronos_1d", c).mean()), 3) for c in CONTROLS},
        "top_minus_bottom_quintile_fwd1_pct": {"raw": round(e3.quintile_spread(df, "kronos_1d", "fwd1") * 100, 4),
                                                "factor_stripped": round(e3.quintile_spread(df, "kronos_1d_resid", "fwd1") * 100, 4)},
        "top_quintile_daily_turnover": round(top_quintile_turnover("kronos_1d_resid"), 3),
        "validation_reference": {"factor_stripped_mean_ic": 0.0138, "t": 1.16, "days": 113},
    }
    df.to_parquet(OUT / "signals.parquet", index=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(stripped_ic.index, stripped_ic.cumsum(), color="#2E43C2", lw=2.2, label="Kronos 1-day, factors stripped")
    ax.plot(raw_ic.index, raw_ic.cumsum(), color="#9CA2AE", lw=1.6, label="Kronos 1-day, raw")
    rev5 = e3.daily_ic(df, "rev5", "fwd1")
    ax.plot(rev5.index, rev5.cumsum(), color="#A86A10", lw=1.4, label="5-day reversal")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_title(f"Test window: cumulative daily RankIC vs. next-day return  ·  verdict {verdict} "
                 f"(stripped IC {stripped['mean_ic']:+.4f}, t {stripped['t_stat_overlap_adjusted']:+.2f})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "test_rankic.png", dpi=130)
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    analyze()
