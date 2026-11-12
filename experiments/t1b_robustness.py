"""T1b: robustness checks on T1's passing signal, from saved signals only (CPU, no new forecasts).

Added after T1 passed, so these can only weaken the claim, never rescue it.
    uv run python -m experiments.t1b_robustness
"""

import json

import numpy as np
import pandas as pd

from experiments import e3_rankic as e3
from experiments import t1_test_1day as t1
from kranos import RESULTS_DIR

OUT = RESULTS_DIR / "t1_test_1day"


def short_term_panel(bars):
    """Very short-horizon controls and execution-lagged targets per (date, ticker)."""
    frames = []
    for t, df in bars.items():
        d = df.set_index("timestamps")
        c, o, h, l = d["close"], d["open"], d["high"], d["low"]
        frames.append(pd.DataFrame({
            "ticker": t,
            "rev1": -(c / c.shift(1) - 1),                       # yesterday's losers bounce
            "rev2": -(c / c.shift(2) - 1),
            "rev3": -(c / c.shift(3) - 1),
            "close_in_range": -((c - l) / (h - l).replace(0, np.nan)),  # closing near the day's low
            "intraday_rev": -(c / o - 1),                        # today's open-to-close move, reversed
            "fwd1_open_to_close": c.shift(-1) / o.shift(-1) - 1,  # enter at tomorrow's open, exit at its close
            "fwd1_skip": c.shift(-2) / c.shift(-1) - 1,          # one full day of delay
        }).rename_axis("date").reset_index())
    return pd.concat(frames, ignore_index=True)


def main():
    _, bars, _ = e3.load_universe()
    df = pd.read_parquet(OUT / "signals.parquet").merge(short_term_panel(bars), on=["date", "ticker"], how="left")

    def s(col, target, h=1):
        return e3.summarize(e3.daily_ic(df, col, target), h)

    wide = t1.CONTROLS + ["rev1", "rev2", "rev3", "close_in_range", "intraday_rev"]
    df["kronos_1d_resid_wide"] = e3.residualize(df, "kronos_1d", wide)

    report = {
        "short_term_factor_ic_fwd1": {c: s(c, "fwd1") for c in ["rev1", "rev2", "rev3", "close_in_range", "intraday_rev"]},
        "rank_corr_kronos_with": {c: round(float(e3.daily_ic(df, "kronos_1d", c).mean()), 3)
                                  for c in ["rev1", "rev2", "rev3", "close_in_range", "intraday_rev"]},
        "kronos_stripped_of_wider_controls": {"controls": wide, "fwd1": s("kronos_1d_resid_wide", "fwd1")},
        "execution_lag": {
            "raw_open_to_close_next_day": s("kronos_1d", "fwd1_open_to_close"),
            "stripped_open_to_close_next_day": s("kronos_1d_resid", "fwd1_open_to_close"),
            "raw_skip_one_day": s("kronos_1d", "fwd1_skip"),
            "stripped_skip_one_day": s("kronos_1d_resid", "fwd1_skip"),
            "rev1_open_to_close_next_day": s("rev1", "fwd1_open_to_close"),
        },
        "quintile_spread_pct_per_day": {
            "raw_close_to_close": round(e3.quintile_spread(df, "kronos_1d", "fwd1") * 100, 4),
            "raw_open_to_close": round(e3.quintile_spread(df, "kronos_1d", "fwd1_open_to_close") * 100, 4),
            "wide_stripped_close_to_close": round(e3.quintile_spread(df, "kronos_1d_resid_wide", "fwd1") * 100, 4),
        },
    }
    (OUT / "robustness.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
