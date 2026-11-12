"""E8: fine-tuned Kronos-small vs. zero-shot on Nifty 100 next-day ranking (see findings.md, E8).

One stage per invocation, chosen by E8_STAGE (required):
    E8_STAGE=train uv run modal run modal_app.py --exp e8_finetune   # fine-tune each GRID run on a T4
    E8_STAGE=val   uv run modal run modal_app.py --exp e8_finetune   # validation forecasts, then decision
    E8_STAGE=test  uv run modal run modal_app.py --exp e8_finetune   # one test run of the chosen variant
Analysis only (CPU, from saved paths):
    uv run python -m experiments.e8_finetune --analyze val|test

Every stage skips work already saved under results/e8_finetune/, so re-running is safe. The test
window is forecast at most once: if test/paths.npy exists, the test stage only re-analyzes.
"""

import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import e3_rankic as e3
from experiments.t1b_robustness import short_term_panel
from kranos import RESULTS_DIR, TEST_START, VAL_END, VAL_START

OUT = RESULTS_DIR / "e8_finetune"
T1_DIR = RESULTS_DIR / "t1_test_1day"  # zero-shot test forecasts (same eval config, seeds 900 + chunk)
STAGES = ("train", "val", "test")

# ---- training grid (fixed before any training run) ----
TRAIN_COMMON = {"model": "small", "lookback": 90, "horizon": 10, "epochs": 4, "steps_per_epoch": 2000,
                "batch_size": 50, "holdout_steps": 200, "seed": 0, "max_minutes": 150}
GRID = {"lr1e-5": {"lr": 1e-5}, "lr4e-5": {"lr": 4e-5}}

# ---- evaluation (identical to T1) ----
EVAL_CFG = {"model": "small", "lookback": 90, "horizon": 1, "n_paths": 10, "T": 0.6, "top_p": 0.9}
N_CHUNKS = 5
VAL_SEED_BASE, TEST_SEED_BASE = 800, 900  # seed = base + chunk index, the same for every variant
ZEROSHOT = "zeroshot"
CONTROLS = ["rev5", "rev10", "mom20", "lowvol20"]                                # T1's 4 controls
WIDE = CONTROLS + ["rev1", "rev2", "rev3", "close_in_range", "intraday_rev"]     # T1b's 9 controls

# ---- pre-registered rules (copied verbatim into the reports) ----
VAL_T_MIN = 1.0
VAL_RULE = ("A fine-tuned variant is eligible if its 9-control-stripped mean RankIC exceeds zero-shot's on the same "
            "(day, stock) pairs AND its 4-control-stripped overlap-adjusted t >= 1.0. Choose the eligible variant with "
            "the highest 9-control-stripped mean RankIC. If none is eligible, chosen = null and E8 stops (no test run).")
TEST_T_MIN = 2.0
TEST_RULE = ("PASS if the chosen fine-tuned variant's 9-control-stripped mean RankIC exceeds zero-shot's on the same "
             "(day, stock) pairs AND its 9-control-stripped t >= 2.0. Otherwise FAIL.")
T1B_REFERENCE = {"zeroshot_test_stripped4": {"mean_ic": 0.0202, "t": 3.96},
                 "zeroshot_test_stripped9": {"mean_ic": 0.0095, "t": 1.93}, "days": 458}


def _stage() -> str:
    stage = os.environ.get("E8_STAGE")
    if stage not in STAGES:
        raise SystemExit(f"E8_STAGE must be one of {STAGES} (got {stage!r}), e.g. "
                         f"E8_STAGE=val uv run modal run modal_app.py --exp e8_finetune")
    return stage


def prepare():
    """Nifty 100 + calendar (E1), plus every Nifty 500 member for training. Cached files return instantly."""
    _stage()  # fail before the slow download if the stage is missing
    from kranos.data import fetch_daily, nifty500

    e3.prepare()
    failures = {}
    for t in nifty500()["ticker"]:
        try:
            fetch_daily(t)
        except Exception as err:  # noqa: BLE001 - a few delisted/renamed symbols are expected
            failures[t] = str(err)
    if failures:
        print(f"nifty500 fetch: {len(failures)} tickers failed (skipped by training): {sorted(failures)}")
    return failures


def orchestrate(forecast_chunk, finetune_run=None):
    stage = _stage()
    if stage == "train":
        if finetune_run is None:
            raise SystemExit("stage train needs finetune_run (run via modal_app.py)")
        run_train(finetune_run)
    elif stage == "val":
        run_val(forecast_chunk)
    else:
        run_test(forecast_chunk)


# ---------------------------------------------------------------- GPU stages (local orchestration) ---

def train_cfg(name: str) -> dict:
    return {**TRAIN_COMMON, **GRID[name], "out_dir": f"/ckpt/e8/{name}"}


def run_train(finetune_run):
    d = OUT / "train"
    d.mkdir(parents=True, exist_ok=True)
    todo = [n for n in GRID if not (d / f"{n}.json").exists()]
    print(f"launching {len(todo)} fine-tuning runs: {todo or 'nothing (all saved)'}")
    if not todo:
        return
    cfgs = [train_cfg(n) for n in todo]
    failed = {}
    for name, cfg, res in zip(todo, cfgs, finetune_run.map(cfgs, order_outputs=True, return_exceptions=True)):
        if isinstance(res, BaseException):
            failed[name] = repr(res)
            print(f"{name} FAILED: {res!r}")
            continue
        (d / f"{name}.json").write_text(json.dumps({**res, "cfg": cfg}, indent=2, default=str))
        print(f"{name} saved -> checkpoint {res.get('checkpoint')}")
    if failed:
        raise RuntimeError(f"fine-tuning failed for {sorted(failed)}; re-run E8_STAGE=train to retry only those")


def _chunks(days: list[str]) -> list[list[str]]:
    return [[str(x) for x in c] for c in np.array_split(days, N_CHUNKS)]  # plain str, not numpy.str_


def _save_forecasts(d, parts, run):
    d.mkdir(parents=True, exist_ok=True)
    paths = np.concatenate([np.load(io.BytesIO(p["paths"])) for p in parts])
    index = pd.concat([pd.read_parquet(io.BytesIO(p["index"])) for p in parts], ignore_index=True)
    np.save(d / "paths.npy", paths)
    index.to_parquet(d / "index.parquet", index=False)
    run = {**run, "gpu": parts[0]["gpu"], "gpu_seconds_total": round(sum(p["gpu_seconds"] for p in parts), 1),
           "gpu_seconds_per_chunk": [round(p["gpu_seconds"], 1) for p in parts], "obs": len(index),
           "days": int(index["date"].nunique()) if len(index) else 0}
    (d / "run.json").write_text(json.dumps(run, indent=2))
    print(d.name, "saved", paths.shape, f"{run['gpu_seconds_total'] / 3600:.2f} GPU-h")


def _checkpoints() -> dict:
    """variant -> checkpoint path (None for zero-shot). Every GRID run must be trained first."""
    missing = [n for n in GRID if not (OUT / "train" / f"{n}.json").exists()]
    if missing:
        raise SystemExit(f"stage val needs every GRID run trained; missing {missing}. Run E8_STAGE=train first.")
    ckpts = {ZEROSHOT: None}
    for n in GRID:
        ck = json.loads((OUT / "train" / f"{n}.json").read_text()).get("checkpoint")
        if not ck:
            raise SystemExit(f"results/e8_finetune/train/{n}.json has no 'checkpoint'")
        ckpts[n] = ck
    return ckpts


def run_val(forecast_chunk):
    ckpts = _checkpoints()
    _, _, cal = e3.load_universe()
    days = [str(d.date()) for d in cal[(cal >= VAL_START) & (cal <= VAL_END)]]
    chunks = _chunks(days)
    todo = [v for v in ckpts if not (OUT / "val" / v / "paths.npy").exists()]
    calls, keys = [], []
    for v in todo:
        for i, c in enumerate(chunks):
            cfg = {**EVAL_CFG, "last_target_day": VAL_END, "seed": VAL_SEED_BASE + i}
            if ckpts[v]:
                cfg["checkpoint"] = ckpts[v]
            calls.append((cfg, c))
            keys.append(v)
    print(f"launching {len(calls)} chunk jobs over {len(days)} validation days for {todo or 'nothing (all saved)'}")
    results = {v: [] for v in todo}
    for v, res in zip(keys, forecast_chunk.starmap(calls, order_outputs=True)):
        results[v].append(res)
    for v, parts in results.items():
        _save_forecasts(OUT / "val" / v, parts, {**EVAL_CFG, "variant": v, "checkpoint": ckpts[v],
                                                 "last_target_day": VAL_END, "seed_base": VAL_SEED_BASE,
                                                 "chunks": N_CHUNKS})
    analyze_val()


def _chosen() -> tuple[str, str]:
    rep = OUT / "val" / "report.json"
    if not rep.exists():
        raise SystemExit("stage test refused: results/e8_finetune/val/report.json does not exist (run E8_STAGE=val)")
    chosen = json.loads(rep.read_text())["decision"]["chosen"]
    if chosen is None:
        raise SystemExit("stage test refused: validation chose no variant (decision.chosen = null), so E8 stops here")
    ck = json.loads((OUT / "train" / f"{chosen}.json").read_text())["checkpoint"]
    return chosen, ck


def run_test(forecast_chunk):
    chosen, ck = _chosen()
    d = OUT / "test"
    if (d / "paths.npy").exists():
        print("test forecasts already saved (the test window is used once); re-analyzing only")
        return analyze_test()
    _, _, cal = e3.load_universe()
    last_day = str(cal.iloc[-1].date())  # last target day; origins stop one bar earlier (as T1)
    days = [str(x.date()) for x in cal[cal >= TEST_START]]
    calls = [({**EVAL_CFG, "last_target_day": last_day, "seed": TEST_SEED_BASE + i, "checkpoint": ck}, c)
             for i, c in enumerate(_chunks(days))]
    print(f"launching {len(calls)} chunk jobs over {len(days)} test days for {chosen} ({ck}), last target {last_day}")
    parts = list(forecast_chunk.starmap(calls, order_outputs=True))
    _save_forecasts(d, parts, {**EVAL_CFG, "variant": chosen, "checkpoint": ck, "last_target_day": last_day,
                               "seed_base": TEST_SEED_BASE, "chunks": N_CHUNKS})
    return analyze_test()


# ---------------------------------------------------------------- analysis (CPU) ---

def _signals(paths: np.ndarray, index: pd.DataFrame, name: str) -> pd.DataFrame:
    """Next-day signal (mean forecast close / close0 - 1) and the min/max of the paths' next-day close."""
    c1 = paths[:, :, 0, 3].astype(np.float64)
    out = index[["date", "ticker", "close0"]].copy()
    out[name] = (c1 / out["close0"].to_numpy()[:, None] - 1).mean(1)
    out[f"{name}_lo"], out[f"{name}_hi"] = c1.min(1), c1.max(1)
    return out


def _intersect(sigs: dict[str, pd.DataFrame], bars) -> pd.DataFrame:
    """Common (date, ticker) pairs across variants, with every control and target attached."""
    names = list(sigs)
    df = sigs[names[0]]
    for n in names[1:]:
        df = df.merge(sigs[n].rename(columns={"close0": f"close0_{n}"}), on=["date", "ticker"], how="inner")
        gap = float(np.max(np.abs(df[f"close0_{n}"] / df["close0"] - 1))) if len(df) else 0.0
        assert gap < 1e-6, f"close0 differs between {names[0]} and {n} by {gap}"
        df = df.drop(columns=f"close0_{n}")
    panel = e3.factor_panel(bars).merge(short_term_panel(bars), on=["date", "ticker"], how="left")
    return df.merge(panel, on=["date", "ticker"], how="left")


def _t(ic: pd.Series) -> float:
    return float(ic.mean() / ic.std() * np.sqrt(len(ic)))  # 1-day horizon: no overlap


def _turnover(df: pd.DataFrame, col: str) -> float:
    """Mean share of the top quintile replaced from one day to the next."""
    sets = df.dropna(subset=[col]).groupby("date").apply(
        lambda x: set(x.loc[x[col].rank(pct=True) > 0.8, "ticker"]), include_groups=False)
    return float(np.mean([1 - len(a & b) / max(len(a), 1) for a, b in zip(sets.iloc[1:], sets.iloc[:-1])]))


def _metrics(df: pd.DataFrame, col: str) -> tuple[dict, dict]:
    """Raw / 4-control / 9-control-stripped next-day RankIC, quintile spreads, calibration of the path range."""
    df[f"{col}_s4"] = e3.residualize(df, col, CONTROLS)
    df[f"{col}_s9"] = e3.residualize(df, col, WIDE)
    cols = {"raw": col, "stripped4": f"{col}_s4", "stripped9": f"{col}_s9"}
    ics = {k: e3.daily_ic(df, c, "fwd1") for k, c in cols.items()}
    ok = df["fwd1"].notna()
    actual = df["close0"] * (1 + df["fwd1"])
    inside = (actual >= df[f"{col}_lo"]) & (actual <= df[f"{col}_hi"])
    n_paths = EVAL_CFG["n_paths"]
    m = {k: e3.summarize(ic, 1) for k, ic in ics.items()}
    m["top_minus_bottom_quintile_fwd1_pct"] = {k: round(e3.quintile_spread(df, c, "fwd1") * 100, 4) for k, c in cols.items()}
    m["calibration_1d"] = {
        "coverage_of_path_range": round(float(inside[ok].mean()), 3),
        "expected_if_calibrated": round((n_paths - 1) / (n_paths + 1), 3),
        "mean_range_width_pct_of_close0": round(float(((df[f"{col}_hi"] - df[f"{col}_lo"]) / df["close0"]).mean()) * 100, 3),
        "mean_abs_actual_1d_return_pct": round(float(df.loc[ok, "fwd1"].abs().mean()) * 100, 3),
    }
    m["rank_corr_with"] = {c: round(float(e3.daily_ic(df, col, c).mean()), 3) for c in WIDE}
    return m, ics


def _paired(a: pd.Series, b: pd.Series) -> dict:
    """Daily IC difference a - b on the days both have, as context for the rules."""
    diff = (a - b).dropna()
    return {"mean_diff": round(float(diff.mean()), 4), "t": round(_t(diff), 2), "days": len(diff),
            "pct_days_a_better": round(float((diff > 0).mean()), 3)}


def _by_quarter(ic: pd.Series) -> dict:
    g = ic.groupby(pd.PeriodIndex(ic.index, freq="Q").astype(str))
    return {k: {"mean_ic": round(float(v.mean()), 4), "days": len(v)} for k, v in g}


def _variants_on_disk() -> list[str]:
    return [v for v in [ZEROSHOT, *GRID] if (OUT / "val" / v / "paths.npy").exists()]


def analyze_val():
    d = OUT / "val"
    variants = _variants_on_disk()
    missing = [v for v in [ZEROSHOT, *GRID] if v not in variants]
    if ZEROSHOT not in variants or len(variants) < 2:
        raise SystemExit(f"analyze_val needs zero-shot and at least one fine-tuned variant; missing {missing}")
    _, bars, _ = e3.load_universe()
    sigs = {v: _signals(np.load(d / v / "paths.npy"), pd.read_parquet(d / v / "index.parquet"), v) for v in variants}
    df = _intersect(sigs, bars)

    report = {"window": {"start": VAL_START, "end": VAL_END}, "eval_cfg": EVAL_CFG,
              "runs": {v: json.loads((d / v / "run.json").read_text()) for v in variants},
              "variants_missing": missing,
              "obs_per_variant_before_intersection": {v: len(s) for v, s in sigs.items()},
              "common_obs": len(df), "days": int(df["date"].nunique()),
              "first_day": str(df["date"].min().date()), "last_day": str(df["date"].max().date()),
              "stocks_per_day": {"min": int(df.groupby("date").size().min()), "max": int(df.groupby("date").size().max())},
              "controls4": CONTROLS, "controls9": WIDE, "variants": {}}
    ics = {}
    for v in variants:
        report["variants"][v], ics[v] = _metrics(df, v)
        if v != ZEROSHOT:
            report["variants"][v]["rank_corr_with_zeroshot"] = round(float(e3.daily_ic(df, v, ZEROSHOT).mean()), 3)
            report["variants"][v]["vs_zeroshot_daily_ic_diff"] = {
                k: _paired(ics[v][k], ics[ZEROSHOT][k]) for k in ["raw", "stripped4", "stripped9"]}

    # Pre-registered decision rule (unrounded numbers; the rounded ones are in "variants").
    zs9 = float(ics[ZEROSHOT]["stripped9"].mean())
    checks, eligible = {}, {}
    for v in variants:
        if v == ZEROSHOT:
            continue
        s9, t4 = float(ics[v]["stripped9"].mean()), _t(ics[v]["stripped4"])
        checks[v] = {"stripped9_mean_ic": s9, "zeroshot_stripped9_mean_ic": zs9, "beats_zeroshot_stripped9": s9 > zs9,
                     "stripped4_t": t4, "stripped4_t_ok": t4 >= VAL_T_MIN}
        if s9 > zs9 and t4 >= VAL_T_MIN:
            eligible[v] = s9
    chosen = max(eligible, key=eligible.get) if eligible else None
    report["decision"] = {"rule": VAL_RULE, "checks": checks, "eligible": sorted(eligible), "chosen": chosen,
                          "next": "E8_STAGE=test runs once on the test window" if chosen else "E8 stops; no test run"}
    df.to_parquet(d / "signals.parquet", index=False)
    (d / "report.json").write_text(json.dumps(report, indent=2, default=str))

    colors = ["#5A6070", "#2E43C2", "#A86A10", "#2F8F6B", "#B5473A"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for v, c in zip(variants, colors):
        s = ics[v]["stripped9"]
        axes[0].plot(s.index, s.cumsum(), color=c, lw=2.2 if v == chosen else 1.6,
                     label=f"{v} ({s.mean():+.4f})" + ("  chosen" if v == chosen else ""))
    axes[0].axhline(0, color="#999", lw=0.8)
    axes[0].set_title(f"Validation: cumulative 9-control-stripped next-day RankIC  ·  chosen: {chosen}", fontsize=10)
    axes[0].legend(fontsize=8)
    x, w = np.arange(len(variants)), 0.27
    for j, (k, c) in enumerate([("raw", "#9CA2AE"), ("stripped4", "#7D8BE0"), ("stripped9", "#2E43C2")]):
        axes[1].bar(x + (j - 1) * w, [report["variants"][v][k]["mean_ic"] for v in variants], w, color=c, label=k)
    axes[1].set_xticks(x, variants, fontsize=9)
    axes[1].axhline(0, color="#999", lw=0.8)
    axes[1].set_title("Mean daily RankIC vs. next-day return", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "val_rankic.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: report[k] for k in ["common_obs", "days", "variants", "decision"]}, indent=2, default=str))
    return report


def _zeroshot_test() -> pd.DataFrame:
    """T1's zero-shot test forecasts as signals; checked against T1's saved signals.parquet."""
    sig = _signals(np.load(T1_DIR / "paths.npy"), pd.read_parquet(T1_DIR / "index.parquet"), ZEROSHOT)
    saved = pd.read_parquet(T1_DIR / "signals.parquet", columns=["date", "ticker", "kronos_1d"])
    chk = sig.merge(saved, on=["date", "ticker"], how="inner")
    gap = float(np.max(np.abs(chk[ZEROSHOT] - chk["kronos_1d"])))
    assert len(chk) == len(sig) and gap < 1e-6, f"T1 paths and signals.parquet disagree (max gap {gap})"
    return sig


def analyze_test():
    d = OUT / "test"
    run = json.loads((d / "run.json").read_text())
    ft = run["variant"]
    _, bars, _ = e3.load_universe()
    sigs = {ft: _signals(np.load(d / "paths.npy"), pd.read_parquet(d / "index.parquet"), ft), ZEROSHOT: _zeroshot_test()}
    df = _intersect(sigs, bars)

    report = {"run": run, "zeroshot_source": str(T1_DIR.relative_to(RESULTS_DIR.parent)),
              "obs_before_intersection": {v: len(s) for v, s in sigs.items()},
              "common_obs": len(df), "days": int(df["date"].nunique()),
              "first_day": str(df["date"].min().date()), "last_day": str(df["date"].max().date()),
              "stocks_per_day": {"min": int(df.groupby("date").size().min()), "max": int(df.groupby("date").size().max())},
              "controls4": CONTROLS, "controls9": WIDE, "variants": {}}
    ics = {}
    for v in [ft, ZEROSHOT]:
        m, ics[v] = _metrics(df, v)
        m["by_quarter"] = {"stripped4": _by_quarter(ics[v]["stripped4"]), "stripped9": _by_quarter(ics[v]["stripped9"])}
        m["execution_lag"] = {f"{k}_{tname}": e3.summarize(e3.daily_ic(df, c, target), 1)
                              for k, c in [("raw", v), ("stripped4", f"{v}_s4"), ("stripped9", f"{v}_s9")]
                              for tname, target in [("open_to_close_next_day", "fwd1_open_to_close"),
                                                    ("skip_one_day", "fwd1_skip")]}
        m["top_quintile_daily_turnover"] = {k: round(_turnover(df, c), 3)
                                            for k, c in [("raw", v), ("stripped4", f"{v}_s4"), ("stripped9", f"{v}_s9")]}
        report["variants"][v] = m
    report["rank_corr_finetuned_with_zeroshot"] = round(float(e3.daily_ic(df, ft, ZEROSHOT).mean()), 3)
    report["finetuned_minus_zeroshot_daily_ic"] = {k: _paired(ics[ft][k], ics[ZEROSHOT][k])
                                                   for k in ["raw", "stripped4", "stripped9"]}

    s9, zs9, t9 = float(ics[ft]["stripped9"].mean()), float(ics[ZEROSHOT]["stripped9"].mean()), _t(ics[ft]["stripped9"])
    verdict = "PASS" if s9 > zs9 and t9 >= TEST_T_MIN else "FAIL"
    report["verdict"] = {"rule": TEST_RULE, "variant": ft, "stripped9_mean_ic": s9, "zeroshot_stripped9_mean_ic": zs9,
                         "beats_zeroshot_stripped9": s9 > zs9, "stripped9_t": t9, "t_ok": t9 >= TEST_T_MIN,
                         "result": verdict}
    report["t1b_reference_all_zeroshot_pairs"] = T1B_REFERENCE
    df.to_parquet(d / "signals.parquet", index=False)
    (d / "report.json").write_text(json.dumps(report, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for v, c in [(ft, "#2E43C2"), (ZEROSHOT, "#5A6070")]:
        for k, ls, lw in [("stripped9", "-", 2.2), ("stripped4", "--", 1.3)]:
            s = ics[v][k]
            ax.plot(s.index, s.cumsum(), color=c, ls=ls, lw=lw, label=f"{v}, {k} ({s.mean():+.4f})")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_title(f"Test: cumulative next-day RankIC, fine-tuned {ft} vs zero-shot  ·  verdict {verdict} "
                 f"(9-stripped {s9:+.4f} vs {zs9:+.4f}, t {t9:+.2f})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "test_rankic.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: report[k] for k in ["common_obs", "days", "finetuned_minus_zeroshot_daily_ic", "verdict"]},
                     indent=2, default=str))
    return report


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--analyze" and sys.argv[2] in ("val", "test"):
        analyze_val() if sys.argv[2] == "val" else analyze_test()
    else:
        raise SystemExit("usage: uv run python -m experiments.e8_finetune --analyze val|test\n" + __doc__)
