#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TOKENIZERS_PARALLELISM=false
CONFIG="${CONFIG:-configs/if_sft_llama2_7b.yaml}"
python -m fpdiag.cli preflight --config "$CONFIG" --quick
python -m fpdiag.cli verify --config "$CONFIG" --quick
python -m fpdiag.cli weight-delta --config "$CONFIG" --stream-shards --quick
python -m fpdiag.cli diagnose --config "$CONFIG" --resume --quick
python -m fpdiag.cli intervene --config "$CONFIG" --resume --quick
python -m fpdiag.cli report --config "$CONFIG"
