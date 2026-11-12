"""Helpers for running Kronos experiments on Indian equities."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Kronos pre-training data ends here; anything later is unseen by the model.
PRETRAIN_CUTOFF = "2024-06-30"
# Used for every tuning choice (temperature, samples, model size, lookback).
VAL_START, VAL_END = "2024-07-01", "2024-12-31"
# Touched only to report a final, already-chosen configuration.
TEST_START = "2025-01-01"
