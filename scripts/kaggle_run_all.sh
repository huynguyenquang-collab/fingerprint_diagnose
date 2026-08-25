#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TOKENIZERS_PARALLELISM=false
CONFIG="${CONFIG:-configs/if_sft_llama2_7b.yaml}"
bash scripts/kaggle_preflight.sh "$@"
python -m fpdiag.cli verify --config "$CONFIG" "$@"
python -m fpdiag.cli weight-delta --config "$CONFIG" --stream-shards "$@"
python -m fpdiag.cli diagnose --config "$CONFIG" --resume "$@"
python -m fpdiag.cli intervene --config "$CONFIG" --resume "$@"
python -m fpdiag.cli report --config "$CONFIG" "$@"
bash scripts/kaggle_package_results.sh
