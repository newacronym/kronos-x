"""E5: pre-registered ablation on validation (see findings.md, E5).

    uv run modal run modal_app.py --exp e5_ablation      # GPU runs for V1, V2, then analysis
    uv run python -m experiments.e5_ablation --analyze    # analysis only (CPU)

B and V3 reuse E3's saved paths; V1 and V2 are new runs, each split across parallel containers.
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
from kranos import RESULTS_DIR, VAL_END, VAL_START

COMMON = {"horizon": 10, "n_paths": 10, "T": 0.6, "top_p": 0.9, "last_target_day": VAL_END}
GPU_VARIANTS = {
    "V1_base": {"model": "base", "lookback": 90},
    "V2_lookback400": {"model": "small", "lookback": 400},
}
N_CHUNKS = 6
SEED = 5
CONTROLS = ["rev5", "rev10", "mom20", "lowvol20"]
T_MIN = 1.0  # pre-registered: a variant must clear this overlap-adjusted t to be chosen
OUT = RESULTS_DIR / "e5_ablation_val"


def prepare():
    e3.prepare()


def orchestrate(forecast_chunk):
    """Runs locally: fan GPU work out across containers, save each variant, then analyze."""
    _, _, cal = e3.load_universe()
    days = [str(d.date()) for d in cal[(cal >= VAL_START) & (cal <= VAL_END)]]
    chunks = [[str(d) for d in c] for c in np.array_split(days, N_CHUNKS)]  # plain str, not numpy.str_

    todo = [n for n in GPU_VARIANTS if not (OUT / n / "paths.npy").exists()]
    calls, keys = [], []
    for name in todo:
        for i, c in enumerate(chunks):
            calls.append(({**COMMON, **GPU_VARIANTS[name], "seed": SEED * 100 + i}, c))
            keys.append(name)
    print(f"launching {len(calls)} chunk jobs for {todo or 'nothing (all variants saved)'}")

    results = {n: [] for n in todo}
    for name, res in zip(keys, forecast_chunk.starmap(calls, order_outputs=True)):
        results[name].append(res)

    for name, parts in results.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        paths = np.concatenate([np.load(io.BytesIO(p["paths"])) for p in parts])
        index = pd.concat([pd.read_parquet(io.BytesIO(p["index"])) for p in parts], ignore_index=True)
        np.save(d / "paths.npy", paths)
        index.to_parquet(d / "index.parquet", index=False)
        run = {**COMMON, **GPU_VARIANTS[name], "seed_base": SEED * 100, "chunks": N_CHUNKS, "gpu": parts[0]["gpu"],
               "gpu_seconds_total": round(sum(p["gpu_seconds"] for p in parts), 1),
               "gpu_seconds_per_chunk": [round(p["gpu_seconds"], 1) for p in parts]}
        (d / "run.json").write_text(json.dumps(run, indent=2))
        print(name, "saved", paths.shape, f"{run['gpu_seconds_total'] / 3600:.2f} GPU-h")
    analyze()


# ---------------------------------------------------------------- analysis (CPU) ---

def _load(name):
    if name == "B":
        d = RESULTS_DIR / "e3_rankic_val"
        return np.load(d / "paths.npz")["paths"], pd.read_parquet(d / "index.parquet")
    d = OUT / name
    return np.load(d / "paths.npy"), pd.read_parquet(d / "index.parquet")


def _signals(paths, index, name):
    r = paths[..., 3] / index["close0"].to_numpy()[:, None, None] - 1
    out = index[["date", "ticker"]].copy()
    out[f"{name}"] = r.mean(axis=(1, 2))                 # mean forecast return over the horizon
    out[f"{name}_step1"] = r[:, :, 0].mean(1)            # next-day forecast return
    out[f"{name}_lo"], out[f"{name}_hi"] = paths[:, :, -1, 3].min(1), paths[:, :, -1, 3].max(1)
    return out


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    uni, bars, _ = e3.load_universe()
    names = ["B"] + list(GPU_VARIANTS)
    sigs = {n: _signals(*_load(n), n) for n in names}

    df = sigs["B"]
    for n in names[1:]:
        df = df.merge(sigs[n], on=["date", "ticker"], how="inner")  # common (day, stock) pairs only
    df = df.merge(e3.factor_panel(bars), on=["date", "ticker"], how="left")
    df["V3_step1"] = df["B_step1"]

    # variant -> (signal column, target, horizon)
    spec = {"B": ("B", "fwd10", 10), "V1_base": ("V1_base", "fwd10", 10),
            "V2_lookback400": ("V2_lookback400", "fwd10", 10), "V3_1day": ("V3_step1", "fwd1", 1)}
    report = {"common_obs": len(df), "days": int(df["date"].nunique()),
              "obs_per_variant_before_intersection": {n: len(s) for n, s in sigs.items()}, "variants": {}}
    cum = {}
    for v, (col, target, h) in spec.items():
        df[f"{col}_resid"] = e3.residualize(df, col, CONTROLS)
        raw, resid = e3.daily_ic(df, col, target), e3.daily_ic(df, f"{col}_resid", target)
        cum[v] = resid
        entry = {"signal": col, "target": target, "raw": e3.summarize(raw, h), "factor_stripped": e3.summarize(resid, h),
                 "rank_corr_with": {c: round(float(e3.daily_ic(df, col, c).mean()), 3) for c in ["rev10", "mom20", "lowvol20"]},
                 "top_minus_bottom_quintile_pct": round(e3.quintile_spread(df, col, target) * 100, 3)}
        if v != "V3_1day":
            base = v if v != "B" else "B"
            ok = df["close10"].notna()
            inside = (df["close10"] >= df[f"{base}_lo"]) & (df["close10"] <= df[f"{base}_hi"])
            entry["coverage_of_10_path_range"] = round(float(inside[ok].mean()), 3)
        report["variants"][v] = entry

    # Pre-registered decision rule.
    eligible = {v: e["factor_stripped"]["mean_ic"] for v, e in report["variants"].items()
                if v != "B" and e["factor_stripped"]["t_stat_overlap_adjusted"] >= T_MIN}
    chosen = max(eligible, key=eligible.get) if eligible else "B"
    report["decision"] = {"rule": f"highest factor-stripped RankIC among variants with t >= {T_MIN}; else B",
                          "eligible": eligible, "chosen_for_test": chosen}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    labels = list(spec)
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, [report["variants"][v]["raw"]["mean_ic"] for v in labels], 0.4, color="#9CA2AE", label="raw")
    axes[0].bar(x + 0.2, [report["variants"][v]["factor_stripped"]["mean_ic"] for v in labels], 0.4, color="#2E43C2", label="factors stripped")
    axes[0].set_xticks(x, labels, fontsize=9)
    axes[0].axhline(0, color="#999", lw=0.8)
    axes[0].set_title("Mean daily RankIC at each variant's horizon (validation)", fontsize=10)
    axes[0].legend(fontsize=8)
    for v, col in zip(labels, ["#5A6070", "#2E43C2", "#A86A10", "#2F8F6B"]):
        axes[1].plot(cum[v].index, cum[v].cumsum(), color=col, lw=1.8, label=v)
    axes[1].axhline(0, color="#999", lw=0.8)
    axes[1].set_title("Cumulative factor-stripped RankIC", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=130)

    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    analyze()
