"""E8 GPU smoke test: a tiny fine-tune on the L4, then forecasts from that checkpoint on a T4.

    uv run modal run modal_app.py --exp e8_smoke
"""

import io
import json
import time

import numpy as np
import pandas as pd

from experiments import e3_rankic as e3
from kranos import RESULTS_DIR, VAL_END

OUT = RESULTS_DIR / "e8_finetune" / "smoke"


def prepare():
    e3.prepare()


def orchestrate(forecast_chunk, finetune_run):
    t0 = time.time()
    # 300 steps at full batch: enough to measure throughput and check fp16 + GradScaler on CUDA.
    m = finetune_run.remote({"lr": 4e-5, "epochs": 1, "steps_per_epoch": 300, "holdout_steps": 40,
                             "out_dir": "/ckpt/e8/smoke"})
    print(json.dumps({k: m[k] for k in ["device", "amp_fp16", "n_tickers", "n_train_windows", "zero_shot_holdout_loss",
                                        "best_holdout_loss", "epochs", "total_seconds", "checkpoint"] if k in m},
                     indent=2, default=str))
    _, _, cal = e3.load_universe()
    days = [str(d.date()) for d in cal[cal >= "2024-07-01"][:2]]
    cfg = {"model": "small", "lookback": 90, "horizon": 1, "n_paths": 10, "T": 0.6, "top_p": 0.9,
           "last_target_day": VAL_END, "seed": 1, "checkpoint": m["checkpoint"]}
    r = forecast_chunk.remote(cfg, days)
    paths, index = np.load(io.BytesIO(r["paths"])), pd.read_parquet(io.BytesIO(r["index"]))
    ret = paths[:, :, 0, 3] / index["close0"].to_numpy()[:, None] - 1
    out = {"train": m, "forecast": {"shape": list(paths.shape), "gpu": r["gpu"], "seconds": round(r["gpu_seconds"], 1),
                                    "mean_abs_1d_forecast_return": float(np.abs(ret.mean(1)).mean()),
                                    "finite": bool(np.isfinite(paths).all())},
           "wall_seconds": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["forecast"], indent=2), "wall", out["wall_seconds"])
