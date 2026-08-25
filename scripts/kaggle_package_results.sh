#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/if_sft_llama2_7b.yaml}"
python -m fpdiag.cli package --config "$CONFIG" "$@"
