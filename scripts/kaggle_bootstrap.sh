#!/usr/bin/env bash
set -euo pipefail

python --version
python - <<'PY'
import importlib.util
for name in ("torch", "transformers", "accelerate", "safetensors"):
    print(f"{name}: {'present' if importlib.util.find_spec(name) else 'missing'}")
PY
python -m pip install --disable-pip-version-check -r requirements-kaggle.txt
python -m pip install --disable-pip-version-check -e . --no-deps
python -c 'import torch, transformers, accelerate, safetensors, fpdiag; print("imports: OK"); print("torch", torch.__version__, "cuda", torch.version.cuda)'
