# IF-SFT LLaMA-2-7B Fingerprint Diagnostics

CLI-first, Kaggle-oriented diagnostics for testing whether the instructional
fingerprint in `cnut1648/LLaMA2-7B-fingerprinted-SFT` is localized and separable
from normal model utility. The exact comparison model is
`NousResearch/Llama-2-7b-hf`.

## Run on Kaggle

Create a Notebook, select **GPU T4 x2**, enable **Internet**, and paste these cells.

### Cell 1 — sanity check

```bash
!nvidia-smi
!df -h /kaggle/working /tmp
```

### Cell 2 — clone

```bash
!git clone https://github.com/huynguyenquang-collab/fingerprint_diagnose.git /kaggle/working/if-sft-weight-diagnostics
%cd /kaggle/working/if-sft-weight-diagnostics
```

### Cell 3 — bootstrap

```bash
!bash scripts/kaggle_bootstrap.sh
```

### Cell 4 — quick test

```bash
!bash scripts/kaggle_run_quick.sh
```

### Cell 5 — full run

```bash
!bash scripts/kaggle_run_all.sh
```

### Cell 6 — package small outputs

```bash
!bash scripts/kaggle_package_results.sh
```

The archive is written to `/kaggle/working/if_sft_fpdiag_results.zip` and excludes
model weights, caches, upstream clones, and large activation dumps.

### One-cell runner

```bash
!git clone https://github.com/huynguyenquang-collab/fingerprint_diagnose.git /kaggle/working/if-sft-weight-diagnostics && \
cd /kaggle/working/if-sft-weight-diagnostics && \
bash scripts/kaggle_bootstrap.sh && \
bash scripts/kaggle_run_all.sh && \
bash scripts/kaggle_package_results.sh
```

## CLI

```bash
python -m fpdiag.cli preflight --config configs/if_sft_llama2_7b.yaml
python -m fpdiag.cli verify --config configs/if_sft_llama2_7b.yaml
python -m fpdiag.cli weight-delta --config configs/if_sft_llama2_7b.yaml --stream-shards
python -m fpdiag.cli diagnose --config configs/if_sft_llama2_7b.yaml --resume
python -m fpdiag.cli intervene --config configs/if_sft_llama2_7b.yaml --resume
python -m fpdiag.cli report --config configs/if_sft_llama2_7b.yaml
```

Every important configuration value can be changed with repeated `--set`, for
example `--set data.n_fp_positive=8 --set paths.scratch_dir=/tmp/fpdiag-alt`.

## Scientific interpretation

The report separates static parameter change, representation diagnostics,
gradient sensitivity, and causal interventions. It compares real keys with
corrupted and OOD-matched controls, reports continuous target likelihood alongside
FSR, and does not infer causality from weight magnitude or CKA.

The upstream published artifact for this exact checkpoint may yield eight unique
positive prompts even though the paper/spec expects ten initially. The pipeline
records the observed count and never fabricates additional keys.

## Sources

- Xu et al., [Instructional Fingerprinting of Large Language Models](https://arxiv.org/abs/2401.12255)
- [Official Model-Fingerprint repository](https://github.com/cnut1648/Model-Fingerprint)
- [Wanda](https://github.com/locuslab/wanda), [ROME](https://github.com/kmeng01/rome), and [Super Weight](https://github.com/mengxiayu/LLMSuperWeight) inspire clearly labeled diagnostics.

Local CPU tests validate numerical and lifecycle primitives. Real scientific
results require the documented Kaggle T4 x2 run; no 7B result is claimed from the
local test suite.
