# IF-SFT LLaMA-2-7B Weight Diagnostics Design

## Purpose

Build a new, CLI-first repository that runs on Kaggle T4 x2 and tests whether the
fingerprint in `cnut1648/LLaMA2-7B-fingerprinted-SFT` is localized in model
components and separable from ordinary language-model utility.

The implementation follows
`IF_SFT_LLAMA2_7B_WEIGHT_LAYER_DIAGNOSTIC_KAGGLE_SPEC.md`. It uses
`NousResearch/Llama-2-7b-hf` as the only valid base checkpoint and the official
`cnut1648/Model-Fingerprint` repository as the source of fingerprint prompts.
It will not train, quantize, or save a modified 7B checkpoint.

## Scope and delivery boundary

The repository will implement all required static, dynamic, intervention,
statistical, plotting, and reporting stages in the source specification. It will
also provide quick and full Kaggle runners, resumable stage execution, CPU unit
tests, and a tiny-model smoke pipeline.

Local completion means that all CPU and tiny-model tests pass and that shell and
CLI entry points validate successfully. Running the real LLaMA-2-7B experiment
requires a Kaggle T4 x2 session with Internet enabled. Any stage not exercised on
that hardware must be described as unverified rather than claimed as tested.

Pushing source code to GitHub is in scope. Publishing a scientifically meaningful
`REPORT.md` is only possible after the real checkpoint run; local tests must never
generate fabricated 7B results.

## Chosen architecture

Use a modular Python package with a resumable stage orchestrator. Each diagnostic
owns one bounded responsibility, consumes explicit artifacts, and emits compact
CSV, Parquet, JSON, or NPZ artifacts. The CLI composes those diagnostics without
placing scientific logic in shell scripts.

This is preferred over a monolithic runner because Kaggle sessions can stop
between expensive phases, model residency must change across phases, and every
intervention needs reliable cleanup. It is preferred over notebooks because the
source specification requires reproducible CLI entry points.

## Repository structure

The repository will contain:

- `src/fpdiag/config.py`: typed configuration, YAML loading, validation, and CLI
  override application.
- `src/fpdiag/cli.py`: public commands `preflight`, `verify`, `weight-delta`,
  `diagnose`, `intervene`, `report`, plus internal stage selection.
- `src/fpdiag/stages.py`: stage registry, dependency checks, state manifests,
  resume behavior, and skip/failure recording.
- `src/fpdiag/environment.py` and `upstream.py`: preflight, provenance, revision,
  disk, connectivity, and upstream source handling.
- `src/fpdiag/data/`: exact fingerprint extraction and deterministic construction
  of key-removed, corrupted-key, OOD-matched, clean-instruction, and clean-LM
  datasets.
- `src/fpdiag/models/`: sequential model loading, LLaMA component mapping,
  generation, teacher-forced inputs, and memory-safe unloading.
- `src/fpdiag/metrics/`: fingerprint endpoints, utility endpoints, CKA, rank
  agreement, bootstrap intervals, and effect-size calculations.
- `src/fpdiag/diagnostics/`: streamed parameter delta, gradients and Fisher,
  activation summaries, logit lens, ablations, patching, candidate ranking,
  reversible restoration, and overlap analysis.
- `src/fpdiag/reporting/`: aggregation, plots, required tables, anti-confound audit,
  and final conclusion logic.
- `src/fpdiag/utils/`: hook contexts, shard resolution, streaming top-K, seeds,
  serialization, and memory guards.
- `scripts/`: safe Kaggle bootstrap, preflight, quick/full runners, and filtered
  result packaging.
- `tests/`: dependency-light unit tests and a tiny causal-LM end-to-end smoke test.

Files will remain focused. Shared interfaces will use serializable dataclasses or
plain records so stage artifacts can be inspected without importing a loaded
model.

## Configuration and CLI

`configs/if_sft_llama2_7b.yaml` will contain every default from the source
specification. Important nested values can be overridden with repeated CLI
arguments such as `--set data.n_fp_positive=8`, while common controls also receive
named flags such as `--output-dir`, `--scratch-dir`, `--seed`, `--start-stage`, and
`--end-stage`.

Configuration loading will:

1. parse YAML;
2. apply typed overrides;
3. reject unknown keys and invalid budgets;
4. resolve paths without redirecting model caches into `/kaggle/working`;
5. persist the effective YAML and its SHA-256 hash.

Public commands match the acceptance tests exactly. All return nonzero on a fatal
failure. A stage can be marked skipped only with a machine-readable reason.

## Stage orchestration and artifacts

The full pipeline uses these numbered stages:

1. `00_preflight`
2. `01_provenance_data`
3. `02_fingerprint_verification`
4. `03_baseline_behavior`
5. `04_static_weight_delta`
6. `05_activation_scan`
7. `06_gradient_fisher`
8. `07_representation_logit_lens`
9. `08_module_ablation`
10. `09_activation_patching`
11. `10_channel_localization`
12. `11_parameter_ranking`
13. `12_targeted_restore`
14. `13_random_controls`
15. `14_overlap_statistics`
16. `15_final_report`
17. `16_package_results`

Each stage writes `outputs/state/stage_NN.json` atomically. The record contains
status, timestamps, config hash, input artifact fingerprints, output list, model
revisions, and skip/failure detail. A `.done` marker is written only after outputs
are validated. `--resume` accepts an existing stage only when its config hash and
declared inputs still match.

Quick mode runs preflight, provenance/data, fingerprint verification, reduced
activation scan, reduced module ablation, sampled streamed delta, and a mini
report. It reduces configured sample counts but does not silently reinterpret a
full-run budget.

## Provenance and data controls

The official repository is shallow-cloned to scratch and never edited in place.
Its commit SHA and the exact source paths used for prompt extraction are recorded.
If old upstream code cannot run, the loader reproduces its construction logic and
regression-tests exact strings against upstream artifacts.

The local related project indicates that the published data for this checkpoint
may expose eight unique verified fingerprint prompts even though the specification
expects ten initially. The implementation will preserve the observed upstream
count, emit a visible count mismatch warning, and never invent extra keys. The
verification gate depends on reproduced behavior, not on coercing the count to ten.

Controls preserve pair identifiers so each positive prompt can be compared with
its key-removed and corrupted forms. OOD-matched strings are deterministically
generated to match token count, non-ASCII fraction, and coarse Unicode script
composition. Generated controls and their seeds are persisted. Clean instruction
and LM samples use fixed IDs and recorded dataset revisions.

## Fingerprint and utility scoring

Teacher forcing concatenates prompt token IDs with target token IDs without
injecting a second BOS token. Prompt labels are `-100`; shifted logits score target
positions only. The scorer emits exact and normalized FSR, sequence and mean-token
log probability, per-token probability/rank/margin, and false-trigger rates.

Utility scoring emits clean NLL, perplexity, and sampled-token KL from the original
fingerprinted model. Baseline logits are cached only for configured positions and
examples. Generated output normalization is explicit and Unicode-preserving.

Fingerprint verification is a hard gate for the full diagnosis. Failure creates
`fingerprint_verification_failed.md`, records available evidence, and prevents
downstream causal claims.

## Memory and model lifecycle

Dynamic stages load the fingerprinted model with FP16, `low_cpu_mem_usage=True`,
`use_cache=False`, and a balanced or explicit two-GPU device map. The effective
dtype and device placement are recorded. Quick mode detects unsupported hardware
and skips stages with reasons instead of attempting an OOM-prone load.

Base and fingerprinted dynamic models are never resident simultaneously. Base
behavior and hidden summaries are computed, moved to compact CPU artifacts, and
the model is unloaded before loading the fingerprinted checkpoint.

Static deltas use Hub metadata and both safetensor index files. The resolver groups
tensors by the required pair of shards, downloads only the current files to
scratch, opens tensors using `safe_open`, computes one compatible tensor at a
time in original dtype-derived precision, writes summaries/top-K, closes mappings,
and deletes eligible temporary shards. It does not materialize either full state
dict.

Gradients are processed per microbatch. Scalar norms, directional products,
Fisher sums, and bounded top-K candidates are transferred to CPU immediately;
dense full-model gradient accumulators are forbidden. CUDA cache clearing occurs
at stage boundaries.

## Static diagnostics

The parameter mapper recognizes embeddings, final norm/head, all 32 blocks, four
attention projections, three MLP projections, and both block norms. Unknown or
shape-incompatible tensors remain in an audit table rather than disappearing.

For each compatible tensor, the static phase computes all specified norms,
relative delta, distribution quantiles, unchanged fraction, cosine similarity,
sign flips, row/column summaries, and global/local top coordinates. Randomized SVD
operates through matrix-vector products or bounded sampled matrices and reports
rank 1/2/4/8/16 energy and effective rank; full SVD of large matrices is rejected.

## Dynamic correlational diagnostics

Gradient scanning computes fingerprint, corrupt, OOD, and clean aggregates using
the correct objective sign and saves raw numerator/denominator values alongside
specificity ratios. Delta-aligned gradient metrics join against streamed base
values only for narrowed tensors or coordinates.

Hook contexts capture configured residual, attention, and MLP sites. Default
storage is aggregate statistics plus selected-token hidden vectors moved to CPU.
Token positions are tracked separately for the key span, last key token, final
input token, and target positions.

Linear CKA uses centered Gram statistics with identity/random sanity checks. The
logit lens applies the model's final normalization convention before `lm_head` and
is labeled correlational, not causal.

## Causal diagnostics

Module ablation applies zero, half-scale, and compatible mean-control replacements
to attention and MLP outputs one module at a time. It always reports fingerprint,
corrupt, and utility changes relative to a cached no-intervention baseline.

Same-model patching caches control activations, injects them into paired positive
runs, and performs the reverse direction where shapes and token alignment permit.
Patch sites and positions are explicit in every result row.

Channel candidates combine separately saved activation, gradient, and delta
components after robust normalization. Top-K interventions use budgets
`1, 4, 16, 32, 64, 128` and matched random seeds.

Weight candidates are narrowed hierarchically by layer, module, channel/axis, then
coordinate. Ranking saves all required raw metrics and never evaluates all 7B
coordinates causally.

Restoration uses a context manager that clones only selected current values,
fetches corresponding base values, patches in place under `no_grad`, evaluates,
and restores exact originals in `finally`. Identical budgets and seeds are used
for random, magnitude, delta, gradient, specificity, interaction, and composite
selectors. Whole-module restoration is optional and guarded by a memory estimate.

## Statistics, plots, and conclusions

Bootstrap sampling is over prompt or paired-prompt IDs with deterministic random
generators. Primary effects receive 95% confidence intervals and effect sizes.
Targeted-versus-random comparisons use identical budgets and bootstrap the
difference in fingerprint-reduction-to-utility-damage ratio.

Reporting joins artifacts defensively, creates every required table, and produces
the eighteen required plots as PNG and PDF with source CSV. Missing prerequisites
create annotated panels or explicit report entries, never zero-filled scientific
results.

The conclusion engine checks predeclared evidence rules and emits exactly one of
`SUPPORTED`, `PARTIALLY_SUPPORTED`, `NOT_SUPPORTED`, or `INCONCLUSIVE`. The report
keeps parameter change, representation difference, gradient sensitivity, and
causal evidence in distinct sections and answers Q1-Q12 explicitly.

## Error handling and safety

Preflight records package versions, CUDA, GPUs, VRAM, RAM, disk, connectivity,
cache paths, git SHA, and config hash before model download. Full mode fails early
without two suitable GPUs, CUDA, adequate scratch space, or LLaMA-2-7B-compatible
architecture metadata.

All hooks, activation patches, and parameter patches are context-managed. Cleanup
and exact rollback are tested on exceptions. Model cache deletion is restricted to
resolved scratch paths owned by the run. Packaging excludes checkpoints, caches,
upstream clones, and large activation dumps.

Shell scripts use strict error handling and propagate nonzero exits. Bootstrap
does not install or downgrade PyTorch/CUDA and installs only missing or
incompatible user-space dependencies.

## Testing strategy

Implementation follows red-green-refactor. CPU tests cover:

- target-only teacher-forced scoring and causal shifting;
- exact/normalized generation scoring;
- config validation and nested overrides;
- LLaMA parameter/module mapping;
- shard index resolution and paired streaming on toy safetensors;
- streaming top-K equality with a naive implementation;
- hook removal on success and exception;
- exact restoration on success and exception;
- CKA identity and independent-random sanity;
- seeded bootstrap determinism;
- stage dependency, resume, stale-config, skip, and failure semantics;
- report generation from synthetic complete and partial artifacts;
- shell script syntax and package exclusion rules.

The tiny-model smoke test uses a tiny public or locally constructible causal LM and
does not assume LLaMA-2 dimensions. Network-dependent smoke coverage is optional in
local CI; an offline locally constructed fixture covers the orchestration path.

## Reuse from local projects

Local code is inspiration, not a dependency. The target scorer structure,
fingerprint provenance lessons, and reversible-gradient patterns may be adapted
from `if_awq_tier0` and `adversarial_quant_if_sft`, with corrected package names,
interfaces, tests, and attribution where appropriate. Quantization implementations
and full dense CPU gradient accumulation are not reused because they violate this
project's scope or Kaggle memory design.

## Acceptance boundary

The repository is source-complete when all specified modules and entry points are
implemented, local tests pass, required documentation and Kaggle cells exist, and
the tested commit is pushed. Real-experiment acceptance additionally requires a
Kaggle T4 x2 run against both exact checkpoints, generation of the packaged result
archive, and review of any runtime deviations. These two acceptance levels will be
reported separately.
