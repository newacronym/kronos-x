# Findings

A running log of what each experiment showed. The newest entries go at the top of the log. The summary is rewritten as conclusions firm up.

## Summary so far

- **Zero-shot Kronos-small has no edge beyond simple factors on daily Nifty 100 data (validation, Jul–Dec 2024).** Its 10-day RankIC of +0.0145 (t = 0.33) is below plain 5-day reversal (+0.028) and far below low volatility (+0.078). Once reversal, momentum and low volatility are stripped out, it's −0.002 (E3).
- **Kronos acts like a reversal / anti-momentum model.** First seen on one stock in E0, confirmed across the cross-section in E3: rank correlation +0.49 with 10-day reversal and −0.57 with 20-day momentum.
- **Bigger model or longer history doesn't fix it (E5).** Kronos-base (4× params) raises raw RankIC only to 0.019, and is still mostly reversal. A 400-bar lookback shifts the signal toward low volatility, and its factor-stripped part is negative.
- **The next-day signal held out of sample (T1, PASS).** On 2025-01-01 → 2026-11-09 (458 days, never touched before), factor-stripped RankIC was +0.020 (t = 3.96), positive in all 8 quarters.
- **But it's mostly known short-term reversal (T1b).**
  - "Closed near the day's low" alone scores +0.043, and plain 1-day reversal +0.026. Kronos's raw signal is +0.025.
  - With those stripped too, Kronos keeps +0.0095 (t = 1.93).
  - The signal lasts one day and is gone if you wait a day.
- **Not tradable as is.** The raw spread is about 0.10% a day with 62% daily turnover, which roughly equals NSE futures-level costs. Kronos's unique part (0.03% a day) is far below them.
- **Bottom line so far:** for ranking Nifty 100 stocks on daily bars, zero-shot Kronos finds real but generic reversal structure that simple rules capture better.
- **At T = 0.6 the sampled paths are overconfident.** The actual 10-day close fell inside the 10-path range 31% of the time, against 82% if calibrated. Volatility and risk work needs a higher temperature (E3).
- **Single-stock point forecasts are poor.** On RELIANCE, the 10-day forecast had a larger error than predicting no change (E0).
- **Power is limited.** Validation has about 11 independent 10-day periods, all in one correction regime. These results rule out a large edge, not a small one.
- **Setup works.** One Nifty 100 day takes about 19 s with Kronos-small at 10 paths on a T4. The full validation run took 36.5 GPU-minutes.
- Engineering: an 8 GB Mac can't fit 1,000 sequences per call, so all runs go to Modal. A T4 handles 1,000 sequences per call with no trouble.

## Conventions

- Every tuning decision uses the validation window (Jul–Dec 2024). The test window (2025 onward) reports chosen configurations only, and each such report is noted here.
- Each entry states the question, the setup, the numbers, what we concluded, and what we do next.
- Any number quoted here comes from a `results/<experiment>/report.json`, which is linked.

---

## Log

### T1 — test run of the next-day signal · pre-registered and run 2026-11-12 · **PASS** (with large caveats, see T1b)

**Question.** Does E5's chosen signal hold on data never looked at? The signal is Kronos-small's next-day forecast with reversal, momentum and low volatility stripped out.

**Setup, fixed in advance.**
- Kronos-small, 90-bar lookback, 10 paths, T = 0.6, top-p = 0.9, 1 forecast step.
- Every trading day from 2025-01-01 → 2026-11-10, scored against the next day's return (the last target is 2026-11-11).
- Same Nifty 100 list. Stocks join once they have 90 bars. Windows crossing the TMPV, TRENT and VEDL demergers are skipped.
- Factors stripped: 5- and 10-day reversal, 20-day momentum, 20-day low volatility. That's the same set as E3 and E5.

**Pass criterion, fixed in advance:** factor-stripped RankIC > 0 with t ≥ 2.0. With a 1-day horizon there's no overlap, so t = mean ÷ std × √days.
- If the validation effect is real and the same size (ICIR ≈ 0.11), about 460 days should give t ≈ 2.3.
- Anything else counts as a fail. Raw RankIC, the quintile spread and turnover get reported for context and are not part of the verdict.

**Run.**
- [report.json](results/t1_test_1day/report.json) · [test_rankic.png](results/t1_test_1day/test_rankic.png) · [signals.parquet](results/t1_test_1day/signals.parquet)
- 458 days (2025-01-01 → 2026-11-09; 2026-11-10 was a market holiday), 95–100 stocks a day, 44,251 forecasts.
- 0.93 T4-hours across 5 parallel chunks, a few minutes of wall time.

**Result: PASS.** Factor-stripped RankIC **+0.0202, t = 3.96** (ICIR 0.185, positive on 58% of days).

| | Test (458 days) | Validation (113 days) |
|---|---|---|
| Factor-stripped RankIC | +0.0202 (t 3.96) | +0.0138 (t 1.16) |
| Raw RankIC | +0.0246 (t 4.25) | +0.0202 (t 1.45) |
| 5-day reversal | +0.0175 (t 2.04) | — |
| 10-day reversal / 20-day momentum / low vol | +0.007 / −0.001 / +0.008 | — |

- **Stable over time:** the factor-stripped RankIC was positive in all 8 quarters, ranging from +0.004 (2026 Q1) to +0.074 (2026 Q4, 26 days).
- **Correlation of the raw signal with factors:** 5-day reversal +0.27, 20-day momentum −0.24.
- **Top-minus-bottom quintile, next day:** +0.10% raw, +0.053% factor-stripped.
- **The top quintile turns over 62% a day.**

### T1b — robustness checks on T1 · 2026-11-12 · done

**Why.** T1's controls included 5- and 10-day reversal but **not 1-day reversal**, or other well-known next-day effects. A next-day signal could just be those. Execution also matters: T1 assumes you trade at the same close the signal uses. These checks were added *after* T1 passed, using saved signals only (no new forecasts), so they can only weaken the claim. [robustness.json](results/t1_test_1day/robustness.json)

**Results (test window, next-day RankIC).**

Simple one-day rules on their own:

| Rule | RankIC (t) | Kronos's rank correlation with it |
|---|---|---|
| 1-day reversal (yesterday's losers) | +0.026 (3.14) | +0.21 |
| 3-day reversal | +0.021 (2.41) | +0.25 |
| **Closing near the day's low** | **+0.043 (5.84)** | +0.28 |
| Reversing today's open-to-close move | +0.027 (3.41) | +0.22 |

| Kronos check | RankIC (t) |
|---|---|
| Stripped of all 9 controls (T1's 4 + the five rules above) | **+0.0095 (1.93)** |
| Enter at tomorrow's open instead of today's close: raw | +0.0175 (3.12) |
| Enter at tomorrow's open: T1-stripped | +0.0124 (2.49) |
| Wait one full day before trading: raw | +0.0024 (0.41) |
| Wait one full day: T1-stripped | +0.0022 (0.43) |

Daily top-minus-bottom quintile spread:

| Version | % per day |
|---|---|
| Raw, close to close | 0.101 |
| Raw, next open to next close | 0.073 |
| Stripped of all 9 controls | 0.029 |

**What we concluded from T1 and T1b together.**
1. **The signal is real.** Kronos's next-day forecast predicts which Nifty 100 stocks do better tomorrow, out of sample: t ≈ 4 over 458 days, positive every quarter. It still works if you enter at the next day's open (t ≈ 3).
2. **Most of it is generic short-term reversal.** A one-line rule, *buy stocks that closed near the day's low*, is about twice as strong (+0.043 against +0.025). Plain 1-day reversal matches Kronos. With those effects stripped as well, Kronos keeps a small edge of its own: **+0.0095, t = 1.93**, just below the usual significance bar.
3. **It lasts one day.** Wait a day and it's gone (+0.002).
4. **It isn't tradable as it stands.**
   - The raw long-short spread is about 0.10% a day gross, 0.07% if you enter at the open, with 62% daily turnover in the top quintile.
   - Replacing 62% of each side every day trades about 1.2× each side's capital daily. At even 0.04–0.05% per trade, futures-level costs on NSE, that eats the whole edge.
   - Kronos's own part, beyond the simple rules, is 0.03% a day, far below any realistic cost.
5. **Answer to the project question, for this use:** zero-shot Kronos does pick up real next-day structure in Indian large caps, but that structure is mostly well-known reversal effects, and simple rules capture them better. As a stock-ranking tool on daily bars, it doesn't justify its cost over a few lines of pandas.

### E5 — ablation on validation · pre-registered and run 2026-11-12 · done

**Question.** Does a bigger model, a longer history, or a shorter horizon give Kronos an edge beyond simple factors?

**Variants.** Everything else is held at E3's settings: 10 paths, T = 0.6, top-p = 0.9, same validation days.

| ID | Change from E3 | Horizon scored | GPU |
|---|---|---|---|
| B | none: E3's Kronos-small, 90-bar lookback (reused, not re-run) | 10d | none |
| V1 | Kronos-base instead of small | 10d | new run |
| V2 | 400-bar lookback instead of 90 | 10d | new run |
| V3 | score the first forecast step against the next-day return | 1d | none: reuses E3's paths |

**Rules, fixed before running.**
- All variants are scored on the same (day, stock) pairs: those where every variant has a forecast. V2 needs 400 bars of history, so a few recent listings drop out.
- **Primary metric:** RankIC of the signal with reversal, momentum and low volatility stripped out (as in E3), at the variant's horizon.
- **Decision:** the variant with the highest primary metric goes to the single test-window run, but only if its overlap-adjusted t ≥ 1.0. If none qualifies, the test run uses baseline B, as a straight replication of the paper's setup.
- Nothing else gets tried on validation after seeing these results. No new variants, temperatures or ensembles get added and then picked.

**Run.**
- [report.json](results/e5_ablation_val/report.json) · [ablation.png](results/e5_ablation_val/ablation.png)
- V1 took 2.05 T4-hours and V2 took 2.71, each split into 6 chunks. Modal ran about 5 containers at a time, so wall time was about 75 minutes.
- Scored on 10,622 common (day, stock) pairs over 113 days. V2's 400-bar requirement dropped 113 pairs from recent listings.

**Results.**

| Variant | Raw RankIC (t) | Factors stripped (t) | Corr. with 10d reversal / 20d momentum / low vol | 10-path range coverage |
|---|---|---|---|---|
| B: small, 90 bars, 10d | +0.0139 (0.31) | −0.0029 (−0.09) | +0.50 / −0.57 / +0.01 | 31% |
| V1: **base**, 90 bars, 10d | +0.0190 (0.37) | +0.0073 (0.18) | +0.48 / −0.57 / +0.06 | 28% |
| V2: small, **400 bars**, 10d | +0.0203 (0.34) | −0.0148 (−0.29) | +0.25 / −0.38 / +0.17 | 27% |
| **V3: small, 90 bars, 1-day** | +0.0202 (**1.45**) | **+0.0138 (1.16)** | +0.31 / −0.25 / +0.02 | — |

**Decision under the pre-registered rule: V3 (the 1-day signal) goes to the test run.** It's the only variant with factor-stripped t ≥ 1.0.

**What we concluded.**
1. **The bigger model barely helps.** Kronos-base lifts raw RankIC from 0.014 to 0.019. It's just as much a reversal / anti-momentum signal (+0.48 / −0.57), and almost nothing is left after stripping factors (+0.007, t = 0.18). Four times the parameters, essentially the same signal.
2. **A longer history changes the character, not the edge.** With 400 bars the signal leans less on reversal (+0.25) and more on low volatility (+0.17). Its raw RankIC looks a bit better, but the factor-stripped part is negative (−0.015). The extra context mostly rediscovers the low-volatility factor.
3. **The 1-day signal is the only lead, and it's weak.**
   - Factor-stripped +0.014 with t = 1.16 over 113 non-overlapping days.
   - With three variants tried, a t of 1.16 is well within what chance alone produces. The test run is there to settle it.
   - **Even if it holds, it's unlikely to be tradable.** The top-minus-bottom quintile spread is just +0.016% per day. A daily-rebalanced long-short book on NSE pays STT, brokerage and slippage of around 0.1% or more per side, about 10× the edge.
4. Every variant's paths are overconfident at T = 0.6 (27–31% coverage against 82% expected). The model size or context length doesn't change that.

**Test run, as pre-registered.**
- V3: Kronos-small, 90-bar lookback, 10 paths, T = 0.6, top-p = 0.9.
- Signal: the mean next-day forecast return, scored against the next-day return.
- Period: 2025-01-01 → latest, factor-stripped RankIC as the main number.
- Only 1 forecast step is needed, so it's cheap: roughly 460 days at about 2–3 s each on a T4. Under autoregressive sampling the first step depends only on the history, so it's the same signal as step 1 of a 10-step forecast.

### E3 — cross-sectional ranking on Nifty 100, validation · 2026-11-12 · done

**Question.** On each day, does Kronos rank Nifty 100 stocks by their next 10-day return better than simple factors? And if it does, is that more than a known factor in disguise?

**Setup.**
- Kronos-small, 10 paths per stock-day, T = 0.6, top-p = 0.9 (the paper's forecasting settings, untuned), 90-day lookback, 10-day horizon.
- 113 origin days (2024-07-04 → 2024-12-16) × 95 stocks = 10,735 forecasts. Origins stop early enough that every 10-day target ends by 2024-12-31, so no test-period price is used.
- Each path was kept, not averaged: [paths.npz](results/e3_rankic_val/paths.npz) holds 10,735 × 10 paths × 10 steps × OHLCVA. Later analyses can reuse it without a GPU.
- Results: [report.json](results/e3_rankic_val/report.json) · [rankic.png](results/e3_rankic_val/rankic.png) · [signals.parquet](results/e3_rankic_val/signals.parquet). GPU time: 36.5 min on a T4.
- Signals:
  - **Kronos:** mean forecast return over the horizon (the paper's signal), forecast return at day 10, share of paths ending up, and mean ÷ spread of paths.
  - **Baselines:** 5- and 10-day reversal, 20-day momentum, 60-day momentum skipping the last 5 days, 20-day low volatility, random.
  - **Kronos with the factors stripped out:** the part of its daily rank not explained by reversal, momentum and low volatility.
  - **Kronos sector-neutral:** the signal minus its industry average each day.

**Results: mean daily RankIC against the 10-day forward return.**

| Signal | RankIC | ICIR | t (overlap-adjusted) | Top − bottom quintile, 10d |
|---|---|---|---|---|
| **Kronos mean (paper's signal)** | **+0.0145** | 0.10 | 0.33 | +0.22% |
| Kronos, day-10 return | +0.0141 | 0.08 | 0.27 | +0.25% |
| Kronos, share of paths up | +0.0125 | 0.09 | 0.28 | +0.25% |
| **Kronos with factors stripped out** | **−0.0018** | −0.02 | −0.05 | — |
| Kronos sector-neutral | +0.0045 | 0.05 | 0.15 | — |
| 5-day reversal | +0.0276 | 0.20 | 0.67 | +0.46% |
| 10-day reversal | +0.0223 | 0.17 | 0.58 | +0.24% |
| 20-day momentum | −0.0143 | −0.09 | −0.30 | −0.12% |
| 20-day low volatility | +0.0776 | 0.35 | 1.17 | +1.00% |
| Random | −0.0005 | 0.00 | −0.01 | −0.14% |

Average daily rank correlation between Kronos's signal and each factor:

| Factor | Correlation |
|---|---|
| 10-day reversal | +0.49 |
| 5-day reversal | +0.40 |
| 20-day momentum | **−0.57** |
| 60-day momentum | −0.32 |
| Low volatility | +0.01 |

Other horizons:
- **5-day forward:** Kronos +0.010, 5-day reversal +0.020, low volatility +0.056.
- **1-day forward:** Kronos +0.014. With factors stripped out it's **+0.013** (t = 1.23). That's the only place Kronos showed anything beyond the factors, and it isn't significant. Low volatility was +0.043 (t = 2.43).

Calibration of the 10-day paths:
- The actual close fell inside the range of the 10 paths **31%** of the time. It should be 82% if the paths were calibrated.
- The paths are far too narrow. Their spread is 2.4%, against a 5.2% cross-sectional spread of actual returns.
- At the same time the mean forecast moves too much: 5.5% average absolute predicted move against 4.6% actual.
- The forecasts were bearish on average: −0.91% predicted against −0.35% actual.

**What we concluded.**
1. **Zero-shot Kronos-small does not beat simple factors on daily Nifty 100 data.** Its ranking signal (+0.0145) is positive but small. It's statistically indistinguishable from zero (t = 0.33) and below plain 5-day reversal.
2. **The E0 hypothesis holds.** Kronos mostly acts as a reversal / anti-momentum signal, correlating +0.49 with 10-day reversal and −0.57 with 20-day momentum. **Once those are stripped out, nothing is left at the 10-day horizon** (−0.002).
3. The faint hint of something extra is at the 1-day horizon (+0.013, t = 1.2). In the cumulative plot, the stripped-out signal climbed steadily until late October and then gave it all back during the Oct–Dec sell-off.
4. **T = 0.6 makes the paths overconfident.** That's harmless for ranking, which only uses the mean, but the paths can't be used for risk or volatility at this temperature. E6 should use T ≈ 0.9–1.0, which the paper used for volatility.
5. **Statistical power is low.** 113 days with a 10-day horizon is about 11 independent periods, all in one regime: a correction, which flatters low volatility. Even low volatility, the strongest signal here, isn't significant at 10 days. This test can rule out a large edge. It can't rule out a small one.

**How this compares with the paper.** The paper reports return-forecasting RankIC of about 0.06 for Kronos-small, but it measures that across many markets and assets with a different setup. Our number is daily cross-sectional ranking within one large-cap universe, so the two aren't directly comparable. Even so, it's well short of the paper's picture.

**Next (to decide).**
- A short ablation on validation (E5), limited to 2–3 changes chosen in advance so as not to overfit 11 independent periods: Kronos-base, a longer 400-bar lookback, and the 1-day horizon.
- Then one test-window run of the chosen configuration, reporting the stripped-out signal as the main number.
- Separately, E6 volatility at T = 0.9–1.0.

### E1 — Nifty 100 daily dataset · 2026-11-12 · done

**Question.** Can we build a clean daily panel for the Nifty 100, and what data problems does it have?

**Setup.**
- NSE's official constituent list, downloaded 2026-11-12 ([data/universe/nifty100_2026-11-12.csv](data/universe/nifty100_2026-11-12.csv)).
- Yahoo daily bars from 2015-01-01, split- and dividend-adjusted.
- The Nifty 50 index (^NSEI) as the trading calendar.
- Results: [report.json](results/e1_data/report.json) · [universe_quality.csv](results/e1_data/universe_quality.csv)

**Results.**
- All 100 members downloaded: 262,801 bars, no impossible OHLC rows. The median history starts 2015-01-01.
- Calendar: 123 trading days in validation and 460 in test (to 2026-11-11).
- **95 of 100 have the 90-bar history needed at the start of validation.** The five that don't are recent listings: HYUNDAI (Oct 2024), ENRIN (Jun 2025), TATACAP (Oct 2025), TMCV (Nov 2025) and VAML (Jun 2026).
- One-day moves above 25% were flagged in 18 tickers. Most are real events: the PSU-bank recapitalization rally on 2017-10-25, the March 2020 COVID crash, Adani in Feb 2023, HINDZINC in May 2024, IRFC in Jan 2024, and many IDEA moves.
- **Four are corporate actions Yahoo didn't adjust:**
  - MOTHERSON +28.6% on 2022-01-14
  - TMPV −40.2% on 2025-10-14 (Tata Motors demerger)
  - TRENT −33.0% on 2026-01-01
  - VEDL −64.9% on 2026-04-30 (Vedanta demerger)

  These are listed in `CORPORATE_ACTION_BREAKS` in [kranos/data.py](kranos/data.py), and any forecast window crossing one is skipped. Three of the four fall in the test window, so this matters for later test runs. None falls inside validation.

**Known biases.**
- **Survivorship.** This is today's index list applied to 2024–2026, so stocks that rose into the index are over-represented. That flatters long-only results and momentum-like signals more than rankings, but it's real. A cleaner version would rebuild membership from NSE's historical index changes.
- **Free data.** Yahoo is free and adjusted, but its corporate-action handling has gaps, as shown above.

**Next.** E3 uses this panel.

### E0 — smoke test · 2026-11-03 → 2026-11-12 · done

**Question.** Does Kronos load and run, how fast is it per Nifty-100-sized batch, is the sampling reproducible, and do forecasts look sane on one stock?

**Setup.**
- RELIANCE.NS daily bars from 2015.
- 100 forecast origins in the validation window, 90-day lookback, 10-day horizon.
- T = 0.6, top-p = 0.9.
- Timing for mini, small and base at 1 and 10 samples.

**First attempt (local Mac, 2026-11-03, abandoned).**
- Data is clean: 2,921 bars from 2015-01-01, longest calendar gap 5 days, no open-vs-previous-close jumps above 10%, no impossible OHLC rows.
- Kronos-mini on the Mac GPU: 100 series × 1 sample took 2.96 s. 100 × 10 samples ran out of memory (9.07 GB limit).
- Yahoo returned the current day's bar while the market was open. It's now filtered out.
- Decision: run all experiments on Modal from here on.

**Modal run (2026-11-12, Tesla T4, torch 2.15.0+cu132).** [report.json](results/e0_smoke/report.json) · [forecasts.png](results/e0_smoke/forecasts.png) · first image build took about 95 s.

*Speed.* 100 series per batch, 10 forecast steps, 1,000 sequences per call. No out-of-memory errors.

| Model | Params | 1 sample | 10 samples |
|---|---|---|---|
| mini | 4.1M | 1.0 s | 3.6 s |
| small | 24.7M | 2.0 s | 14.6 s |
| base | 102.3M | 6.4 s | 59.4 s |

Projected cost of a daily Nifty 100 run at 10 samples (validation is about 125 trading days, validation + test about 590):

| Model | Validation only | Validation + test |
|---|---|---|
| mini | ~8 min | ~35 min |
| small | ~30 min | ~2.4 h |
| base | ~2 h | ~9.7 h |

Base on the full period is the only expensive case. Options for it: rebalance weekly (5× cheaper), use a faster GPU, or add a KV cache to the generation loop. The upstream code re-runs the whole context at every step.

*Determinism.* The same seed gives bit-identical forecasts. Different seeds move the 10-sample mean forecast by 0.37% on average. That's small next to a typical 10-day move of 3.5%, so 10 samples is enough to stabilize the signal.

*Sanity preview.* RELIANCE.NS, Kronos-small, 10 samples, 100 overlapping origins from 2024-07-01 to 2024-11-22. Only 31% of the 10-day windows ended up, because the stock fell over this period.

| Measure | Kronos | Comparison |
|---|---|---|
| 10-day return error (MAE) | 5.75% | 3.46% if you predict no change |
| Average size of predicted move | 4.27% | 3.46% actual |
| Direction hit rate | 34% | 31% for "always up", 69% for "always down" |
| Rank correlation with the actual 10-day return | +0.16 | −0.28 for 20-day momentum |

**What we concluded.**
1. The pipeline is ready for multi-stock runs. Kronos-small at 10 samples is the affordable default. Base gets checked on validation in E5.
2. As a point forecast for one stock, Kronos is worse than predicting no change. Its moves are too large and often in the wrong direction over this falling period. That doesn't settle much on its own: the windows overlap heavily (about 10 independent periods), and the paper's claim is about ranking stocks, not single-stock accuracy.
3. The plot suggests mean reversion. After the sharp drop into 2024-10-04 it forecast a rebound, which didn't come. After the rally into 2024-07-01 it forecast a pullback, which didn't come either. Its signal is positively correlated with actual returns while momentum's is negative, which fits a reversal pattern.

**Next.**
- E1: build the Nifty 100 daily dataset.
- E3: measure Kronos's daily cross-sectional RankIC on validation against momentum, short-term reversal (5-day and 10-day) and random rankings. Also measure how correlated Kronos's signal is with plain reversal, to see whether it adds anything beyond it.
- Later: when the paths are kept rather than averaged, check whether Kronos's predicted spread is too wide.
