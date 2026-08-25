import json

import pytest

safetensors = pytest.importorskip("safetensors.torch")
torch = pytest.importorskip("torch")

from fpdiag.utils.shard_io import paired_safe_open, resolve_tensor_shard


def test_resolves_and_opens_toy_shards(tmp_path):
    for prefix, value in (("base", 1.0), ("fp", 2.0)):
        root = tmp_path / prefix; root.mkdir()
        safetensors.save_file({"weight": torch.tensor([value])}, root / "part.safetensors")
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "part.safetensors"}}))
    base = resolve_tensor_shard(tmp_path / "base/model.safetensors.index.json", "weight")
    fp = resolve_tensor_shard(tmp_path / "fp/model.safetensors.index.json", "weight")
    with paired_safe_open(base, fp) as (left, right):
        assert (right.get_tensor("weight") - left.get_tensor("weight")).item() == 1
