from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path


def load_weight_map(index_path):
    return json.loads(Path(index_path).read_text())["weight_map"]


def resolve_tensor_shard(index_path, tensor_name):
    mapping = load_weight_map(index_path)
    if tensor_name not in mapping: raise KeyError(tensor_name)
    return str(Path(index_path).parent / mapping[tensor_name])


@contextmanager
def paired_safe_open(base_shard, fp_shard):
    from safetensors import safe_open
    with safe_open(str(base_shard), framework="pt", device="cpu") as base, \
         safe_open(str(fp_shard), framework="pt", device="cpu") as fp:
        yield base, fp
