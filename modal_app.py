"""Run an experiment on Modal GPUs and bring its results back to results/.

    uv run modal run modal_app.py --exp e0_smoke

Flow: the experiment's prepare() runs locally (CPU only, downloads data) and data/ is synced to
the `kranos-data` volume. Then either
  - main() runs remotely on one GPU and its results folder comes back as a tarball, or
  - if the experiment defines orchestrate(forecast_chunk), that runs locally and fans GPU work
    out across parallel containers via forecast_chunk.starmap(...).
"""

import importlib
import inspect
import io
import os
import sys
import tarfile
import time
from pathlib import Path

import modal

PROJECT = "/root/project"
GPU = os.environ.get("KRANOS_GPU", "T4")

app = modal.App("kranos")
hf_cache = modal.Volume.from_name("kranos-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("kranos-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("kranos-ckpt", create_if_missing=True)  # fine-tuned weights (E8)
CKPT = "/ckpt"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.2", "numpy<2.1", "pandas==2.2.2", "einops==0.8.1", "huggingface_hub==0.33.1",
        "safetensors==0.6.2", "tqdm==4.67.1", "matplotlib==3.9.3", "scipy>=1.11", "pyarrow>=15",
        "yfinance>=0.2.50",
    )
    .env({"HF_HOME": "/cache/hf", "PYTHONPATH": PROJECT})
    .add_local_dir("kranos", f"{PROJECT}/kranos")
    .add_local_dir("experiments", f"{PROJECT}/experiments")
    .add_local_dir("vendor/Kronos/model", f"{PROJECT}/vendor/Kronos/model")
)
gpu_fn = dict(image=image, gpu=GPU, cpu=4, memory=16384, timeout=4 * 3600,
              volumes={"/cache": hf_cache, f"{PROJECT}/data": data_vol, CKPT: ckpt_vol})
TRAIN_GPU = os.environ.get("KRANOS_TRAIN_GPU", "T4")


def _enter_project():
    os.chdir(PROJECT)
    sys.path.insert(0, PROJECT)


@app.function(**gpu_fn)
def run_experiment(module: str) -> bytes:
    _enter_project()
    out_dir = importlib.import_module(module).main()
    hf_cache.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out_dir, arcname=Path(out_dir).name)
    return buf.getvalue()


@app.function(**gpu_fn)
def forecast_chunk(cfg: dict, days: list[str]) -> dict:
    """Kronos paths for one block of days under one configuration."""
    _enter_project()
    import numpy as np
    import pandas as pd
    import torch

    from kranos.data import load_cached, nifty100
    from kranos.forecast import forecast_days
    from kranos.kronos_api import best_device, load_predictor

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = best_device()
    bars = load_cached(list(nifty100()["ticker"]))
    if cfg.get("checkpoint"):  # a fine-tuned predictor saved by finetune_run (path inside /ckpt)
        ckpt_vol.reload()
    pred = load_predictor(cfg["model"], device, checkpoint=cfg.get("checkpoint"))
    t0 = time.perf_counter()
    paths, index = forecast_days(pred, bars, [pd.Timestamp(d) for d in days], lookback=cfg["lookback"],
                                 horizon=cfg["horizon"], n_paths=cfg["n_paths"], T=cfg["T"], top_p=cfg["top_p"],
                                 last_target_day=cfg["last_target_day"], log_every=5)
    hf_cache.commit()
    p_buf, i_buf = io.BytesIO(), io.BytesIO()
    np.save(p_buf, paths)
    index.to_parquet(i_buf, index=False)
    return {"paths": p_buf.getvalue(), "index": i_buf.getvalue(), "gpu_seconds": time.perf_counter() - t0,
            "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else device}


@app.function(**{**gpu_fn, "gpu": TRAIN_GPU, "timeout": 6 * 3600})
def finetune_run(cfg: dict) -> dict:
    """Fine-tune one predictor (E8). kranos.finetune.train writes checkpoints under cfg["out_dir"] in /ckpt."""
    _enter_project()
    from kranos.finetune import train

    result = train(cfg)
    ckpt_vol.commit()
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(exp: str):
    module = f"experiments.{exp}"
    mod = importlib.import_module(module)
    if hasattr(mod, "prepare"):
        mod.prepare()

    with data_vol.batch_upload(force=True) as batch:
        batch.put_directory("data", "/")
    print(f"synced data/ to volume kranos-data; running {module} on {GPU}")

    if hasattr(mod, "orchestrate"):
        extra = {"finetune_run": finetune_run} if "finetune_run" in inspect.signature(mod.orchestrate).parameters else {}
        mod.orchestrate(forecast_chunk, **extra)
        return

    blob = run_experiment.remote(module)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        tar.extractall("results", filter="data")
    print("results unpacked into results/")
