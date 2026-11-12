"""E1: build the Nifty 100 daily dataset since 2015 and check its quality.

CPU only, runs locally:  uv run python -m experiments.e1_data
"""

import json

import numpy as np
import pandas as pd

from kranos import RESULTS_DIR, TEST_START, VAL_END, VAL_START
from kranos.data import fetch_daily, nifty100, quality_report

LOOKBACK, HORIZON = 90, 10
OUT = RESULTS_DIR / "e1_data"
# A one-day move this large in a Nifty 100 stock is far more often an unadjusted bonus or split
# than a real move. Flag for manual review rather than silently dropping.
SUSPECT_MOVE = 0.25


def prepare():
    """Download every member, then the Nifty 50 index (^NSEI) as the trading calendar."""
    failures = {}
    for t in nifty100()["ticker"]:
        try:
            fetch_daily(t)
        except Exception as err:  # noqa: BLE001 - record and continue; reported in main()
            failures[t] = str(err)
    fetch_daily("^NSEI")
    return failures


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    failures = prepare()
    uni = nifty100()
    cal = fetch_daily("^NSEI")["timestamps"]
    val_days = cal[(cal >= VAL_START) & (cal <= VAL_END)]
    test_days = cal[cal >= TEST_START]

    rows = []
    for _, u in uni.iterrows():
        if u.ticker in failures:
            rows.append({"ticker": u.ticker, "industry": u.industry, "error": failures[u.ticker]})
            continue
        df = fetch_daily(u.ticker)
        q = quality_report(df)
        ret = df["close"].pct_change().abs()
        suspects = df.loc[ret > SUSPECT_MOVE, "timestamps"].dt.date.astype(str).tolist()
        dates = set(df["timestamps"])
        bars_before = lambda d: int((df["timestamps"] < d).sum())
        rows.append({
            "ticker": u.ticker, "industry": u.industry, **q,
            "suspect_moves": suspects,
            "bars_before_val": bars_before(VAL_START),
            "bars_before_test": bars_before(TEST_START),
            "missing_val_days": int(sum(d not in dates for d in val_days)),
            "missing_test_days": int(sum(d not in dates for d in test_days)),
            "eligible_val_start": bars_before(VAL_START) >= LOOKBACK,
            "eligible_test_start": bars_before(TEST_START) >= LOOKBACK,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "universe_quality.csv", index=False)

    ok = tab[tab.get("error").isna()] if "error" in tab else tab
    report = {
        "universe_file": "data/universe/nifty100_2026-11-12.csv",
        "members": len(uni),
        "downloaded": int(len(ok)),
        "failed": failures,
        "calendar": {"source": "^NSEI", "val_days": int(len(val_days)), "test_days": int(len(test_days)),
                     "last_day": str(cal.iloc[-1].date())},
        "history_start_median": str(pd.to_datetime(ok["first"]).median().date()),
        "eligible_at_val_start": int(ok["eligible_val_start"].sum()),
        "eligible_at_test_start": int(ok["eligible_test_start"].sum()),
        "late_listings": ok.loc[~ok["eligible_val_start"], ["ticker", "first"]].to_dict("records"),
        "tickers_with_suspect_moves": ok.loc[ok["suspect_moves"].str.len() > 0, ["ticker", "suspect_moves"]].to_dict("records"),
        "tickers_missing_over_5_val_days": ok.loc[ok["missing_val_days"] > 5, ["ticker", "missing_val_days"]].to_dict("records"),
        "bad_ohlc_rows_total": int(ok["bad_ohlc_rows"].sum()),
        "rows_total": int(ok["rows"].sum()),
        "industries": uni["industry"].value_counts().to_dict(),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print(json.dumps(report, indent=2, default=str))
    return OUT


if __name__ == "__main__":
    main()
