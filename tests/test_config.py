from pathlib import Path

import pytest

from fpdiag.config import ConfigError, load_config


CONFIG = Path(__file__).parents[1] / "configs" / "if_sft_llama2_7b.yaml"


def test_exact_models_and_nested_typed_overrides():
    cfg = load_config(CONFIG, ["data.n_fp_positive=8", "dynamic.use_cache=false"])
    assert cfg.models.fingerprinted == "cnut1648/LLaMA2-7B-fingerprinted-SFT"
    assert cfg.models.base == "NousResearch/Llama-2-7b-hf"
    assert cfg.data.n_fp_positive == 8
    assert cfg.dynamic.use_cache is False


def test_unknown_override_is_rejected():
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(CONFIG, ["data.missing=1"])


def test_config_hash_is_stable():
    assert load_config(CONFIG).config_hash == load_config(CONFIG).config_hash
