# kronos-x

Does [Kronos](https://github.com/shiyu-coder/Kronos) ([paper](https://arxiv.org/abs/2508.02739)), an open foundation model for candlestick data, give a usable forecasting edge on Indian stocks?

Tested zero-shot on Nifty 100 daily bars by ranking stocks cross-sectionally, as in the paper. Full write-up: [findings.md](findings.md). Visual explainer of the paper: [docs/kronos-explainer.html](docs/kronos-explainer.html).

## Result

Next-day RankIC on the test window, 2025–26 (458 days):

| Signal | Zero-shot | Fine-tuned on Nifty 500 |
|---|---|---|
| Kronos-small, raw | +0.025 (t 4.25) | +0.032 (t 5.23) |
| With reversal, momentum and low-vol removed | +0.020 (t 3.96) | +0.028 (t 5.27) |
| With 1-day reversal effects also removed | +0.0095 (t 1.93) | **+0.0137 (t 2.75)** |
| "Closed near the day's low" rule, for comparison | +0.043 | +0.043 |

Both versions passed their pre-registered test.

- **Zero-shot (T1):** the signal is real, but it is mostly well-known short-term reversal.
- **Fine-tuning (E8):** trained on Nifty 500 data before Jul 2024, it raises every measure. Its day-by-day gain over zero-shot is not significant, though.
- **What both share:** the signal lasts one day and turns over about 65% of the top quintile daily, so it doesn't survive NSE trading costs.

The 10-day forecasts from the paper's setup showed no edge on validation, whatever the model size or lookback.

## Method

- **Splits, fixed up front:** data before Jul 2024 is never evaluated (Kronos may have seen it in pre-training). Jul–Dec 2024 is validation for every choice. 2025 onward is test, used once.
- **Signal:** mean of 10 sampled forecast paths ÷ today's close − 1, ranked across stocks daily.
- **Scoring:** daily Spearman RankIC, overlap-adjusted t-stats, and residuals after removing known factors.

| ID | Experiment | Outcome |
|---|---|---|
| E0 | Smoke test: speed, determinism | T4: small model ≈ 15 s per 100 stocks × 10 paths |
| E1 | Nifty 100 data since 2015 | 100/100 stocks, 4 corporate-action breaks handled |
| E3 | 10-day RankIC, validation | +0.015 raw, ≈ 0 after factors |
| E5 | Ablation: base model, 400-bar lookback, 1-day horizon | 1-day chosen |
| T1 | One run on the test window | Pass (see above) |
| E8 | Fine-tune the predictor on Nifty 500 before Jul 2024 (33 T4-min) | Pass; gain over zero-shot not significant |

## Run

```bash
uv sync
uv run modal run modal_app.py --exp e3_rankic       # GPU work on a Modal T4
uv run python -m experiments.t1b_robustness         # CPU-only analysis
E8_STAGE=train uv run modal run modal_app.py --exp e8_finetune   # then E8_STAGE=val, then E8_STAGE=test
```

Each experiment's `prepare()` downloads Yahoo Finance bars locally. `main()` or `orchestrate()` then runs on Modal, and the reports come back to `results/`. You need a Modal account (`modal setup`). Large forecast arrays aren't committed; rerun an experiment to regenerate them.

```
kranos/          data download, Kronos loading, batched path sampling
experiments/     one script per experiment
modal_app.py     Modal GPU runner
vendor/Kronos/   upstream model code (MIT, commit 67b630e)
results/         reports and plots
```

Research only, not investment advice. The universe is today's Nifty 100, so there's survivorship bias.
