"""E8: fine-tune the Kronos predictor on Indian daily bars (Nifty 500, pre-cutoff only).

Mirrors upstream's recipe (vendor/Kronos/finetune/train_predictor.py, dataset.py, config.py):
frozen pre-trained tokenizer, windows of lookback + horizon + 1 bars, per-window normalization
from the lookback rows, clip 5, next-token loss on both token streams via `head.compute_loss`,
AdamW(0.9, 0.95, wd 0.1), OneCycleLR(pct_start 0.03, div_factor 10), grad-norm clip 3.0, and
random windows drawn each epoch instead of a full pass.

Normalization and the amount column
-----------------------------------
* amount = volume * mean(open, high, low, close). Upstream's fine-tune preprocessing
  (qlib_data_preprocess.py) and KronosPredictor (when no amount column is given, which is how
  kranos calls it) both compute it this way, so training sees exactly what inference sees.
* Training (upstream dataset.py): mean/std (ddof 0) over the first `lookback` rows of the window,
  applied to all rows (lookback and the future targets), x = (x - mean) / (std + 1e-5), clip +-5.
* Inference (KronosPredictor.predict_batch): mean/std over the whole context it is handed (the
  history only), same eps, clip 5. That is the same rule -- stats from past rows only -- with one
  difference: at inference the stats cover however long a context the caller passes (e.g. 512
  bars), while training uses `lookback` (90) rows, as upstream does. We follow upstream training.
  Because stats are per window, they cannot be precomputed per ticker; we precompute the raw
  float32 feature/time arrays once and normalize each batch with vectorized numpy.

The s2 stream: upstream calls `model(s1_in, s2_in, stamp)` without teacher forcing, so the s2 head
is conditioned on an s1 token *sampled* from the model's own s1 logits (as at inference). We keep
that, which makes the s2 loss (and the total) slightly stochastic; the holdout is evaluated under
a fixed torch seed, and s1/s2 losses are reported separately (s1 is deterministic).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kranos import PRETRAIN_CUTOFF
from kranos.data import CORPORATE_ACTION_BREAKS, load_cached, nifty500
from kranos.kronos_api import MODELS, Kronos, KronosTokenizer, best_device

FEATURES = ["open", "high", "low", "close", "volume", "amount"]
MAX_ABS_RETURN = 0.35  # |close/prev_close - 1| above this inside a window -> likely unadjusted action

DEFAULTS = {
    "model": "small",
    "lookback": 90,
    "horizon": 10,
    "lr": 4e-5,
    "epochs": 4,
    "steps_per_epoch": 2000,
    "batch_size": 50,
    "holdout_steps": 200,
    "seed": 0,
    "max_minutes": 150,
    "train_end": PRETRAIN_CUTOFF,
    "holdout_start": "2023-07-01",
    "out_dir": None,  # required
    "max_tickers": None,
    "device": None,  # None -> kranos.kronos_api.best_device() (CUDA if present, else CPU)
    "clip": 5.0,
    "weight_decay": 0.1,
    "betas": [0.9, 0.95],
    "grad_clip": 3.0,
    "log_every": 100,
    "amp": None,  # None -> fp16 autocast + GradScaler on CUDA, fp32 elsewhere
}


# ----------------------------------------------------------------------------- data


class WindowData:
    """All tickers' bars concatenated into flat float32 arrays plus the valid window starts.

    A window is `window` consecutive bars of one ticker, identified by its global start row, so a
    batch is a single fancy-index gather.
    """

    def __init__(self, bars: dict[str, pd.DataFrame], window: int, holdout_start: str, train_end: str):
        hs, te = np.datetime64(pd.Timestamp(holdout_start)), np.datetime64(pd.Timestamp(train_end))
        feats, stamps, train, hold = [], [], [], []
        skipped = {"bad_bar": 0, "corporate_action_break": 0, "jump_gt_35pct": 0, "straddles_holdout_start": 0}
        tickers, offset = [], 0
        for t, df in bars.items():
            df = df.sort_values("timestamps")
            df = df[df["timestamps"] <= pd.Timestamp(train_end)].reset_index(drop=True)
            n = len(df)
            if n < window:
                continue
            ts = df["timestamps"].to_numpy(dtype="datetime64[ns]")
            px = df[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
            vol = df["volume"].to_numpy(dtype=np.float64)
            x = np.column_stack([px, vol, vol * px.mean(axis=1)])

            bad = ~np.isfinite(x).all(axis=1) | (vol <= 0) | (px <= 0).any(axis=1)
            close = px[:, 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = np.abs(close[1:] / close[:-1] - 1)
            jump = np.concatenate([[False], ~(ret <= MAX_ABS_RETURN)])  # NaN counts as a jump
            jump &= ~bad & ~np.concatenate([[False], bad[:-1]])  # pairs with a bad bar count as bad_bar

            starts = np.arange(n - window + 1)
            ends = starts + window - 1
            cbad = np.concatenate([[0], np.cumsum(bad)])
            cjump = np.concatenate([[0], np.cumsum(jump)])
            has_bad = (cbad[ends + 1] - cbad[starts]) > 0
            has_jump = (cjump[ends + 1] - cjump[starts + 1]) > 0  # returns strictly inside the window
            # Same rule as kranos.forecast.build_windows: skip if a break date is within [first, last].
            has_ca = np.zeros(len(starts), dtype=bool)
            for b in CORPORATE_ACTION_BREAKS.get(t, []):
                b = np.datetime64(pd.Timestamp(b))
                has_ca |= (ts[starts] <= b) & (b <= ts[ends])

            is_train = ts[ends] < hs
            is_hold = ts[starts] >= hs  # ends <= train_end by the truncation above
            skipped["straddles_holdout_start"] += int((~is_train & ~is_hold).sum())
            use = is_train | is_hold
            skipped["bad_bar"] += int((use & has_bad).sum())
            skipped["corporate_action_break"] += int((use & ~has_bad & has_ca).sum())
            skipped["jump_gt_35pct"] += int((use & ~has_bad & ~has_ca & has_jump).sum())
            ok = ~has_bad & ~has_ca & ~has_jump
            train.append(starts[is_train & ok] + offset)
            hold.append(starts[is_hold & ok] + offset)

            tsi = pd.DatetimeIndex(ts)
            # Upstream time features: minute, hour, weekday, day, month (minute/hour are 0 for daily bars).
            stamps.append(np.column_stack([tsi.minute, tsi.hour, tsi.weekday, tsi.day, tsi.month]).astype(np.float32))
            feats.append(np.nan_to_num(x).astype(np.float32))  # bad rows are never inside a used window
            tickers.append(t)
            offset += n

        self.window = window
        self.tickers = tickers
        self.x = np.concatenate(feats) if feats else np.zeros((0, 6), np.float32)
        self.stamp = np.concatenate(stamps) if stamps else np.zeros((0, 5), np.float32)
        self.train_starts = np.concatenate(train) if train else np.zeros(0, np.int64)
        self.holdout_starts = np.concatenate(hold) if hold else np.zeros(0, np.int64)
        self.skipped = skipped

    def batch(self, starts: np.ndarray, lookback: int, clip: float):
        """(normalized x (B, window, 6), time stamps (B, window, 5)) as float32 torch tensors."""
        idx = starts[:, None] + np.arange(self.window)[None, :]
        x = self.x[idx]
        past = x[:, :lookback]
        mean, std = past.mean(axis=1, keepdims=True), past.std(axis=1, keepdims=True)
        x = np.clip((x - mean) / (std + 1e-5), -clip, clip).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(self.stamp[idx])


# ----------------------------------------------------------------------------- training


def _loss(model, tokenizer, x, stamp, device, amp):
    """Upstream's step: frozen tokenizer (fp32, so tokens match fp32 inference), shift by one."""
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x, half=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1, :])
        return model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])


def _evaluate(model, tokenizer, data, starts, cfg, device, amp):
    """Mean holdout loss over fixed batches; torch RNG fixed so the sampled-s1 noise is repeatable."""
    model.eval()
    bs = cfg["batch_size"]
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
    torch.manual_seed(cfg["seed"] + 12345)
    tot = s1_tot = s2_tot = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(starts), bs):
            x, stamp = data.batch(starts[i:i + bs], cfg["lookback"], cfg["clip"])
            loss, l1, l2 = _loss(model, tokenizer, x.to(device), stamp.to(device), device, amp)
            tot, s1_tot, s2_tot, n = tot + loss.item(), s1_tot + l1.item(), s2_tot + l2.item(), n + 1
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    model.train()
    return {"loss": tot / n, "s1": s1_tot / n, "s2": s2_tot / n} if n else {"loss": float("nan"), "s1": float("nan"), "s2": float("nan")}


def _write(out_dir: Path, result: dict):
    tmp = out_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, default=str))
    os.replace(tmp, out_dir / "metrics.json")


def load_data(cfg: dict) -> WindowData:
    tickers = list(nifty500()["ticker"])
    if cfg.get("max_tickers"):
        tickers = tickers[: cfg["max_tickers"]]
    bars = load_cached(tickers)
    window = cfg["lookback"] + cfg["horizon"] + 1  # upstream: lookback + predict + 1 (one shift for targets)
    return WindowData(bars, window, cfg["holdout_start"], cfg["train_end"])


def count_windows(cfg: dict | None = None) -> dict:
    """Window counts only (no model, no training)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    data = load_data(cfg)
    return {"n_tickers": len(data.tickers), "n_train_windows": int(len(data.train_starts)),
            "n_holdout_windows": int(len(data.holdout_starts)), "skipped": data.skipped}


def train(cfg: dict) -> dict:
    cfg = {**DEFAULTS, **cfg}
    if not cfg["out_dir"]:
        raise ValueError("cfg['out_dir'] is required")
    if pd.Timestamp(cfg["train_end"]) > pd.Timestamp(PRETRAIN_CUTOFF):
        raise ValueError(f"train_end {cfg['train_end']} is after PRETRAIN_CUTOFF {PRETRAIN_CUTOFF}")
    if pd.Timestamp(cfg["holdout_start"]) > pd.Timestamp(cfg["train_end"]):
        raise ValueError("holdout_start must not be after train_end")
    device = cfg["device"] or best_device()
    if device.startswith("mps") and os.environ.get("KRANOS_ALLOW_MPS") != "1":
        raise ValueError("MPS is disabled (set KRANOS_ALLOW_MPS=1 to allow)")
    cfg["device"] = device
    amp = device.startswith("cuda") if cfg["amp"] is None else bool(cfg["amp"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    budget = cfg["max_minutes"] * 60 if cfg["max_minutes"] else math.inf

    torch.manual_seed(cfg["seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])
    data = load_data(cfg)
    n_train, n_hold = len(data.train_starts), len(data.holdout_starts)
    print(f"tickers={len(data.tickers)} train_windows={n_train} holdout_windows={n_hold} skipped={data.skipped}", flush=True)
    if n_train == 0 or n_hold == 0:
        raise ValueError("no train or holdout windows")
    hold_rng = np.random.default_rng(cfg["seed"] + 1)
    n_hold_use = min(cfg["holdout_steps"] * cfg["batch_size"], n_hold)
    hold_starts = np.sort(hold_rng.choice(data.holdout_starts, size=n_hold_use, replace=False))

    model_id, tok_id, _ = MODELS[cfg["model"]]
    tokenizer = KronosTokenizer.from_pretrained(tok_id).to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    model = Kronos.from_pretrained(model_id).to(device)
    model.train()
    device_name = torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else device

    result = {
        "cfg": cfg, "device_name": device_name, "amp_fp16": amp,
        "n_tickers": len(data.tickers), "n_train_windows": int(n_train), "n_holdout_windows": int(n_hold),
        "n_holdout_windows_used": int(n_hold_use), "window_bars": data.window, "skipped": data.skipped,
        "epochs": [], "zero_shot_holdout_loss": None, "zero_shot_holdout": None,
        "best_epoch": None, "best_holdout_loss": None, "beats_zero_shot": None,
        "checkpoint": None, "stopped_early": False, "total_seconds": None,
    }

    t0 = time.perf_counter()
    zs = _evaluate(model, tokenizer, data, hold_starts, cfg, device, amp)
    eval_seconds = time.perf_counter() - t0
    result["zero_shot_holdout"], result["zero_shot_holdout_loss"] = zs, zs["loss"]
    print(f"epoch 0 (pre-trained) holdout loss={zs['loss']:.4f} s1={zs['s1']:.4f} s2={zs['s2']:.4f} "
          f"eval {eval_seconds:.0f}s", flush=True)
    result["total_seconds"] = time.perf_counter() - t_start
    _write(out_dir, result)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], steps_per_epoch=cfg["steps_per_epoch"],
                                                    epochs=cfg["epochs"], pct_start=0.03, div_factor=10)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = math.inf
    bs, log_every = cfg["batch_size"], cfg["log_every"]

    for epoch in range(1, cfg["epochs"] + 1):
        rng = np.random.default_rng(cfg["seed"] * 10000 + epoch)  # upstream: fresh seeded draw per epoch
        draws = rng.integers(0, n_train, size=cfg["steps_per_epoch"] * bs)
        t_epoch, run, run_n, ep_sum, steps = time.perf_counter(), 0.0, 0, 0.0, 0
        for step in range(cfg["steps_per_epoch"]):
            # Leave time for this epoch's holdout pass (about as long as the zero-shot one).
            if time.perf_counter() - t_start > budget - 1.5 * eval_seconds - 30:
                result["stopped_early"] = True
                print(f"time budget reached at epoch {epoch} step {step}", flush=True)
                break
            x, stamp = data.batch(data.train_starts[draws[step * bs:(step + 1) * bs]], cfg["lookback"], cfg["clip"])
            loss, _, _ = _loss(model, tokenizer, x.to(device, non_blocking=True),
                               stamp.to(device, non_blocking=True), device, amp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            li = loss.item()
            if not math.isfinite(li):
                print(f"non-finite loss at epoch {epoch} step {step + 1}", flush=True)
            run, run_n, ep_sum, steps = run + li, run_n + 1, ep_sum + li, steps + 1
            if (step + 1) % log_every == 0 or step + 1 == cfg["steps_per_epoch"]:
                el = time.perf_counter() - t_epoch
                print(f"epoch {epoch}/{cfg['epochs']} step {step + 1}/{cfg['steps_per_epoch']} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} loss={run / run_n:.4f} "
                      f"{(step + 1) / el:.2f} it/s total {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
                run, run_n = 0.0, 0
        if steps == 0:
            break
        ho = _evaluate(model, tokenizer, data, hold_starts, cfg, device, amp)
        rec = {"epoch": epoch, "steps": steps, "train_loss": ep_sum / steps, "holdout_loss": ho["loss"],
               "holdout_s1": ho["s1"], "holdout_s2": ho["s2"], "minutes": (time.perf_counter() - t_epoch) / 60}
        result["epochs"].append(rec)
        print(f"epoch {epoch} train={rec['train_loss']:.4f} holdout={ho['loss']:.4f} (s1 {ho['s1']:.4f} s2 {ho['s2']:.4f}) "
              f"zero-shot={zs['loss']:.4f} {rec['minutes']:.1f} min", flush=True)
        if ho["loss"] < best:  # upstream: best over fine-tuned epochs (init inf), not vs zero-shot
            best = ho["loss"]
            model.save_pretrained(str(out_dir / "best"))
            result.update(best_epoch=epoch, best_holdout_loss=best, checkpoint=str(out_dir / "best"),
                          beats_zero_shot=bool(best < zs["loss"]))
            print(f"saved {out_dir / 'best'}", flush=True)
        result["total_seconds"] = time.perf_counter() - t_start
        _write(out_dir, result)
        if result["stopped_early"]:
            break

    result["total_seconds"] = time.perf_counter() - t_start
    _write(out_dir, result)
    return result


SMOKE = {"max_tickers": 5, "batch_size": 4, "steps_per_epoch": 3, "epochs": 2, "holdout_steps": 2,
         "device": "cpu", "log_every": 1, "max_minutes": 20}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E8 Kronos fine-tuning")
    ap.add_argument("--out-dir", help="output directory (checkpoint in <out-dir>/best)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: 5 tickers, batch 4, 3 steps x 2 epochs")
    ap.add_argument("--count", action="store_true", help="only count windows for the full universe")
    ap.add_argument("--cfg", default="{}", help="JSON dict of cfg overrides")
    a = ap.parse_args()
    over = json.loads(a.cfg)
    if a.count:
        print(json.dumps(count_windows(over), indent=2))
    else:
        if not a.out_dir:
            ap.error("--out-dir is required")
        cfg = {**(SMOKE if a.smoke else {}), **over, "out_dir": a.out_dir}
        print(json.dumps(train(cfg), indent=2, default=str))
