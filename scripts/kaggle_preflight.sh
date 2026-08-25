#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
df -h /kaggle/working /tmp
python -m fpdiag.cli preflight --config configs/if_sft_llama2_7b.yaml "$@"
