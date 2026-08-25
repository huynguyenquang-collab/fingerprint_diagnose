from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
from pathlib import Path


EXPECTED_LLAMA2_7B = {"hidden_size": 4096, "intermediate_size": 11008, "num_hidden_layers": 32,
                      "num_attention_heads": 32, "vocab_size": 32000}


class PreflightError(RuntimeError):
    pass


def validate_llama_config(config):
    mismatch = {key: {"expected": value, "actual": config.get(key)} for key, value in EXPECTED_LLAMA2_7B.items()
                if config.get(key) != value}
    if mismatch:
        raise PreflightError(f"checkpoint is not LLaMA-2-7B-like: {mismatch}")


def collect_environment(config_hash=None):
    packages = {}
    for name in ("torch", "transformers", "accelerate", "safetensors"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    gpu = []
    cuda = False
    try:
        import torch
        cuda = torch.cuda.is_available()
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu.append({"index": index, "name": props.name, "vram_bytes": props.total_memory})
    except ImportError: pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda_available": cuda, "gpus": gpu, "cpu_ram_bytes": _ram(),
            "disk": {path: shutil.disk_usage(path)._asdict() for path in ("/tmp", "/kaggle/working") if Path(path).exists()},
            "hf_home": os.environ.get("HF_HOME"), "git_commit": _git_sha(), "config_hash": config_hash}


def _ram():
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError: return None


def _git_sha():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_full_hardware(environment):
    if not environment["cuda_available"]: raise PreflightError("CUDA is unavailable")
    if len(environment["gpus"]) < 2: raise PreflightError("full mode requires two GPUs")
    # GPU vendors label T4 as 16 GB using decimal units. PyTorch reports its
    # usable capacity as 15,835,660,288 bytes (~14.75 GiB), so a 15 GiB binary
    # threshold incorrectly rejects the reference Kaggle T4 x2 accelerator.
    minimum_16gb_class_bytes = 15_000_000_000
    undersized = [gpu for gpu in environment["gpus"][:2]
                  if gpu["vram_bytes"] < minimum_16gb_class_bytes]
    if undersized:
        measured = ", ".join(f"{gpu.get('name', 'GPU')}={gpu['vram_bytes'] / 1e9:.2f} GB"
                             for gpu in undersized)
        raise PreflightError(f"full mode requires two 16 GB-class GPUs; undersized: {measured}")
