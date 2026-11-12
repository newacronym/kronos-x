"""Thin wrapper around the vendored Kronos code (vendor/Kronos, upstream commit 67b630e)."""

import os
import sys

import numpy as np
import torch

from kranos import ROOT

sys.path.insert(0, str(ROOT / "vendor" / "Kronos"))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

MODELS = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}


def best_device() -> str:
    """CUDA when present (Modal), else CPU. The Mac GPU is used only with KRANOS_ALLOW_MPS=1."""
    if torch.cuda.is_available():
        return "cuda:0"
    if os.environ.get("KRANOS_ALLOW_MPS") == "1" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_oom(err: Exception) -> bool:
    return isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()


def _free(device: str):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def load_predictor(size: str = "small", device: str | None = None, checkpoint: str | None = None) -> KronosPredictor:
    """Pre-trained predictor, or a fine-tuned one from `checkpoint` (a save_pretrained directory).

    The tokenizer is always the pre-trained one: fine-tuning (E8) keeps it frozen.
    """
    model_id, tok_id, ctx = MODELS[size]
    tokenizer = KronosTokenizer.from_pretrained(tok_id)
    model = Kronos.from_pretrained(checkpoint or model_id)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device=device or best_device(), max_context=ctx)


def n_params(predictor: KronosPredictor) -> int:
    return sum(p.numel() for p in predictor.model.parameters())


def predict_chunked(predictor: KronosPredictor, xs, xts, yts, pred_len, sample_count, max_seqs=1000, **kw):
    """predict_batch in chunks so that series x samples per call stays under max_seqs.

    Halves the chunk and retries on out-of-memory. (On an 8 GB M3, 1,000 sequences in one call
    exhausted MPS memory even for Kronos-mini.) Returns (predictions, sequences per call used).
    """
    while True:
        per_call = max(1, max_seqs // sample_count)
        try:
            out = []
            for i in range(0, len(xs), per_call):
                out += predictor.predict_batch(xs[i:i + per_call], xts[i:i + per_call], yts[i:i + per_call],
                                               pred_len=pred_len, sample_count=sample_count, verbose=False, **kw)
                _free(predictor.device)
            return out, per_call * sample_count
        except (RuntimeError, torch.OutOfMemoryError) as err:
            if not _is_oom(err) or per_call == 1:
                raise
            _free(predictor.device)
            max_seqs //= 2


def predict_paths(predictor: KronosPredictor, xs, xts, yts, pred_len, n_paths, max_seqs=1000, **kw):
    """Every sampled path, not just their mean: array (series, n_paths, pred_len, OHLCVA).

    Upstream predict() averages its samples internally. Repeating each series n_paths times with
    sample_count=1 draws from the same distribution and keeps each path separate.
    """
    rep = lambda items: [v for v in items for _ in range(n_paths)]
    preds, seqs = predict_chunked(predictor, rep(xs), rep(xts), rep(yts), pred_len, sample_count=1,
                                  max_seqs=max_seqs, **kw)
    arr = np.stack([p.to_numpy(dtype=np.float32) for p in preds])
    return arr.reshape(len(xs), n_paths, pred_len, arr.shape[-1]), seqs
