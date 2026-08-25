import pytest

from fpdiag.environment import PreflightError, validate_full_hardware, validate_llama_config


def test_exact_llama_architecture_required():
    validate_llama_config({"hidden_size": 4096, "intermediate_size": 11008, "num_hidden_layers": 32,
                           "num_attention_heads": 32, "vocab_size": 32000})
    with pytest.raises(PreflightError):
        validate_llama_config({"hidden_size": 1})


def test_full_preflight_accepts_two_real_nvidia_t4_devices():
    environment = {
        "cuda_available": True,
        "gpus": [
            {"name": "Tesla T4", "vram_bytes": 15_835_660_288},
            {"name": "Tesla T4", "vram_bytes": 15_835_660_288},
        ],
    }
    validate_full_hardware(environment)
