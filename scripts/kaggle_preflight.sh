#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
df -h /kaggle/working /tmp
CONFIG="${CONFIG:-configs/if_sft_llama2_7b.yaml}"
python -m fpdiag.cli preflight --config "$CONFIG" "$@"
