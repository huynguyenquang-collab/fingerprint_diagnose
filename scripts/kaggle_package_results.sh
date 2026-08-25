#!/usr/bin/env bash
set -euo pipefail
OUTPUT="${OUTPUT:-/kaggle/working/if_sft_fpdiag}"
ARCHIVE="/kaggle/working/if_sft_fpdiag_results.zip"
test -d "$OUTPUT"
cd "$OUTPUT"
zip -q -r "$ARCHIVE" . \
  -x '*huggingface*' '*checkpoints*' '*upstream*' '*.safetensors' '*.bin' '*activation_dump*'
echo "$ARCHIVE"
