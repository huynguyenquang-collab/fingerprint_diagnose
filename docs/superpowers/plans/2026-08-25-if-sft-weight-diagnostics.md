# IF-SFT LLaMA-2-7B Weight Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, resumable, Kaggle-first CLI repository for static, dynamic, and causal localization of the released IF-SFT LLaMA-2-7B fingerprint.

**Architecture:** A typed configuration and stage-orchestration core coordinates small diagnostic modules that communicate through inspectable artifacts. Static analysis streams paired safetensor shards; dynamic analysis loads one FP16 model at a time; interventions are reversible context managers; report generation distinguishes missing, correlational, gradient, and causal evidence.

**Tech Stack:** Python 3.9+, PyTorch, Transformers, Accelerate, Hugging Face Hub, safetensors, NumPy, pandas, SciPy, scikit-learn, matplotlib, PyYAML, pytest, Bash.

---

### Task 1: Package, configuration, and reproducibility core

**Files:**
- Create: `pyproject.toml`
- Create: `requirements-kaggle.txt`
- Create: `configs/if_sft_llama2_7b.yaml`
- Create: `src/fpdiag/__init__.py`
- Create: `src/fpdiag/config.py`
- Create: `src/fpdiag/utils/seeds.py`
- Create: `src/fpdiag/utils/io.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_nested_override_is_typed(default_config_path):
    cfg = load_config(default_config_path, ["data.n_fp_positive=8", "dynamic.use_cache=false"])
    assert cfg.data.n_fp_positive == 8
    assert cfg.dynamic.use_cache is False

def test_unknown_override_is_rejected(default_config_path):
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(default_config_path, ["data.missing=1"])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL because `fpdiag.config` does not exist.

- [ ] **Step 3: Implement package metadata, exact YAML defaults, typed recursive records, overrides, config hashing, atomic JSON/YAML output, and deterministic seeding**

```python
def apply_override(raw: dict[str, Any], expression: str) -> None:
    dotted, encoded = expression.split("=", 1)
    cursor: Any = raw
    for key in dotted.split(".")[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise ConfigError(f"unknown configuration key: {dotted}")
        cursor = cursor[key]
    leaf = dotted.split(".")[-1]
    if not isinstance(cursor, dict) or leaf not in cursor:
        raise ConfigError(f"unknown configuration key: {dotted}")
    cursor[leaf] = yaml.safe_load(encoded)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_config.py -q`
Expected: all configuration tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements-kaggle.txt configs src/fpdiag tests/test_config.py
git commit -m "feat: add configuration and package core"
```

### Task 2: Stage state, resume, and CLI routing

**Files:**
- Create: `src/fpdiag/stages.py`
- Create: `src/fpdiag/cli.py`
- Test: `tests/test_stages.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for atomic state, stale config detection, ranges, and public commands**

```python
def test_resume_rejects_stale_config(tmp_path):
    store = StageStore(tmp_path, "hash-a")
    store.complete("00_preflight", outputs=[])
    assert StageStore(tmp_path, "hash-b").can_resume("00_preflight") is False

@pytest.mark.parametrize("command", ["preflight", "verify", "weight-delta", "diagnose", "intervene", "report"])
def test_public_command_has_help(command):
    assert cli.main([command, "--help"]) == 0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_stages.py tests/test_cli.py -q`
Expected: FAIL on missing stage and CLI modules.

- [ ] **Step 3: Implement the stage registry, dependency validation, atomic JSON plus `.done` writes, `failed/skipped` reasons, and argparse subcommands**

```python
@contextmanager
def stage_run(store: StageStore, name: str):
    store.start(name)
    try:
        yield
    except StageSkipped as exc:
        store.skip(name, str(exc))
    except Exception as exc:
        store.fail(name, f"{type(exc).__name__}: {exc}")
        raise
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_stages.py tests/test_cli.py -q`
Expected: PASS.

```bash
git add src/fpdiag/stages.py src/fpdiag/cli.py tests/test_stages.py tests/test_cli.py
git commit -m "feat: add resumable stage CLI"
```

### Task 3: Environment preflight and upstream provenance

**Files:**
- Create: `src/fpdiag/environment.py`
- Create: `src/fpdiag/upstream.py`
- Create: `src/fpdiag/utils/memory.py`
- Test: `tests/test_checkpoint_identity.py`
- Test: `tests/test_environment.py`

- [ ] **Step 1: Write failing architecture and full-mode preflight tests**

```python
def test_llama2_identity_accepts_exact_shape():
    validate_llama_config({"hidden_size": 4096, "intermediate_size": 11008,
        "num_hidden_layers": 32, "num_attention_heads": 32, "vocab_size": 32000})

def test_full_mode_requires_two_t4_sized_gpus(fake_environment):
    with pytest.raises(PreflightError, match="two GPUs"):
        validate_full_hardware(fake_environment(gpus=[{"vram_bytes": 16 << 30}]))
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_checkpoint_identity.py tests/test_environment.py -q`
Expected: FAIL because preflight is absent.

- [ ] **Step 3: Implement dependency-light environment collection, disk guards, exact architecture checks, Hub revision resolution, shallow upstream clone, and provenance persistence**

```python
EXPECTED_LLAMA2_7B = {"hidden_size": 4096, "intermediate_size": 11008,
    "num_hidden_layers": 32, "num_attention_heads": 32, "vocab_size": 32000}

def validate_llama_config(config: Mapping[str, Any]) -> None:
    mismatches = {k: (EXPECTED_LLAMA2_7B[k], config.get(k)) for k in EXPECTED_LLAMA2_7B
                  if config.get(k) != EXPECTED_LLAMA2_7B[k]}
    if mismatches:
        raise PreflightError(f"checkpoint is not LLaMA-2-7B-like: {mismatches}")
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_checkpoint_identity.py tests/test_environment.py -q`
Expected: PASS.

```bash
git add src/fpdiag/environment.py src/fpdiag/upstream.py src/fpdiag/utils/memory.py tests
git commit -m "feat: add Kaggle preflight and provenance"
```

### Task 4: Fingerprint/control datasets and scoring

**Files:**
- Create: `src/fpdiag/data/__init__.py`
- Create: `src/fpdiag/data/fingerprint.py`
- Create: `src/fpdiag/data/controls.py`
- Create: `src/fpdiag/data/utility.py`
- Create: `src/fpdiag/models/generation.py`
- Create: `src/fpdiag/metrics/fingerprint.py`
- Create: `src/fpdiag/metrics/utility.py`
- Test: `tests/test_target_scoring.py`
- Test: `tests/test_controls.py`

- [ ] **Step 1: Write failing tests for target masking, Unicode scoring, exact upstream extraction, pair preservation, and deterministic OOD controls**

```python
def test_target_tokens_only_are_scored():
    result = score_target_logits(logits, [-100, -100, 3, 4])
    assert result.token_ids == [3, 4]

def test_ood_controls_are_deterministic_and_do_not_contain_key(tokenizer):
    first = make_ood_matched("秘密-key", tokenizer, seed=42)
    assert first == make_ood_matched("秘密-key", tokenizer, seed=42)
    assert first != "秘密-key"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_target_scoring.py tests/test_controls.py -q`
Expected: FAIL on missing modules.

- [ ] **Step 3: Implement exact/normalized FSR, target sequence metrics, generation, upstream JSONL discovery/extraction, count-mismatch provenance, paired controls, and fixed utility samples**

```python
shift_logits, shift_labels = logits[..., :-1, :], labels[..., 1:]
mask = shift_labels.ne(-100)
active = shift_logits[mask]
targets = shift_labels[mask]
logp = active.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_target_scoring.py tests/test_controls.py -q`
Expected: PASS.

```bash
git add src/fpdiag/data src/fpdiag/models/generation.py src/fpdiag/metrics tests
git commit -m "feat: add fingerprint data controls and metrics"
```

### Task 5: Model lifecycle and LLaMA mapping

**Files:**
- Create: `src/fpdiag/models/__init__.py`
- Create: `src/fpdiag/models/loader.py`
- Create: `src/fpdiag/models/llama_map.py`
- Test: `tests/test_llama_module_map.py`
- Test: `tests/test_model_loader.py`

- [ ] **Step 1: Write failing mapping and load-argument tests**

```python
def test_projection_mapping():
    ref = parse_parameter_name("model.layers.17.mlp.down_proj.weight")
    assert (ref.layer, ref.family, ref.projection) == (17, "mlp", "down_proj")

def test_dynamic_load_kwargs_are_memory_safe():
    kwargs = dynamic_load_kwargs("float16")
    assert kwargs["device_map"] == "balanced"
    assert kwargs["low_cpu_mem_usage"] is True
```

- [ ] **Step 2: Verify RED, implement sequential context-managed loading and mapping, verify GREEN, commit**

Run RED/GREEN: `python -m pytest tests/test_llama_module_map.py tests/test_model_loader.py -q`
Expected GREEN: PASS.

```bash
git add src/fpdiag/models tests/test_llama_module_map.py tests/test_model_loader.py
git commit -m "feat: add memory-safe model loading and mapping"
```

### Task 6: Streamed exact weight-delta diagnostics

**Files:**
- Create: `src/fpdiag/utils/shard_io.py`
- Create: `src/fpdiag/utils/topk.py`
- Create: `src/fpdiag/diagnostics/__init__.py`
- Create: `src/fpdiag/diagnostics/weight_delta.py`
- Create: `src/fpdiag/diagnostics/layer_delta.py`
- Test: `tests/test_shard_streaming.py`
- Test: `tests/test_topk.py`
- Test: `tests/test_weight_delta.py`

- [ ] **Step 1: Write failing toy-safetensor tests for resolver, one-pair-at-a-time iteration, top-K equality, metrics, and randomized-SVD rejection guard**

```python
def test_streaming_topk_equals_naive():
    values = torch.tensor([1.0, -7.0, 3.0, 9.0])
    stream = StreamingTopK(2)
    stream.update(values[:2], offset=0)
    stream.update(values[2:], offset=2)
    assert stream.indices.tolist() == [3, 1]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_shard_streaming.py tests/test_topk.py tests/test_weight_delta.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement index resolution, Hub shard acquisition abstraction, paired `safe_open`, tensor metrics, row/column summaries, global coordinate top-K, grouped aggregation, and randomized SVD**

```python
with safe_open(base_path, framework="pt", device="cpu") as base_file, \
     safe_open(fp_path, framework="pt", device="cpu") as fp_file:
    for name in names:
        base = base_file.get_tensor(name)
        fp = fp_file.get_tensor(name)
        yield name, base, fp
        del base, fp
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_shard_streaming.py tests/test_topk.py tests/test_weight_delta.py -q`
Expected: PASS.

```bash
git add src/fpdiag/utils src/fpdiag/diagnostics tests
git commit -m "feat: add streamed exact weight delta diagnostics"
```

### Task 7: Hooks, activation summaries, CKA, and logit lens

**Files:**
- Create: `src/fpdiag/utils/hooks.py`
- Create: `src/fpdiag/diagnostics/activations.py`
- Create: `src/fpdiag/diagnostics/activation_outliers.py`
- Create: `src/fpdiag/diagnostics/logit_lens.py`
- Create: `src/fpdiag/metrics/cka.py`
- Test: `tests/test_hooks_cleanup.py`
- Test: `tests/test_cka.py`
- Test: `tests/test_activations.py`

- [ ] **Step 1: Write failing hook exception-cleanup, selected-position, and CKA sanity tests**

```python
def test_hooks_removed_after_exception(module):
    with pytest.raises(RuntimeError):
        with capture_outputs({"x": module}):
            raise RuntimeError("boom")
    assert not module._forward_hooks

def test_linear_cka_identity():
    x = np.random.default_rng(42).normal(size=(32, 16))
    assert linear_cka(x, x) == pytest.approx(1.0)
```

- [ ] **Step 2: Verify RED, implement context-managed hooks and compact statistics/hidden captures, CKA, final-norm logit lens, verify GREEN, commit**

Run: `python -m pytest tests/test_hooks_cleanup.py tests/test_cka.py tests/test_activations.py -q`
Expected GREEN: PASS.

```bash
git add src/fpdiag/utils/hooks.py src/fpdiag/diagnostics src/fpdiag/metrics/cka.py tests
git commit -m "feat: add activation CKA and logit-lens diagnostics"
```

### Task 8: Gradient, Fisher, and delta-alignment scans

**Files:**
- Create: `src/fpdiag/diagnostics/gradients.py`
- Create: `src/fpdiag/diagnostics/empirical_fisher.py`
- Test: `tests/test_gradients.py`

- [ ] **Step 1: Write failing tests proving per-example clearing, scalar aggregation, correct signed alignment, and no weight mutation**

```python
def test_gradient_scan_clears_grads_and_preserves_weights(tiny_model, batch):
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    scan_gradients(tiny_model, [batch], objective=target_objective)
    assert all(p.grad is None for p in tiny_model.parameters())
    assert all(torch.equal(before[n], p) for n, p in tiny_model.named_parameters())
```

- [ ] **Step 2: Verify RED, implement streaming norms/Fisher/top-K and delta-gradient joins, verify GREEN, commit**

Run: `python -m pytest tests/test_gradients.py -q`
Expected GREEN: PASS.

```bash
git add src/fpdiag/diagnostics/gradients.py src/fpdiag/diagnostics/empirical_fisher.py tests/test_gradients.py
git commit -m "feat: add streaming gradient and Fisher diagnostics"
```

### Task 9: Reversible module, activation, channel, and parameter interventions

**Files:**
- Create: `src/fpdiag/diagnostics/module_ablation.py`
- Create: `src/fpdiag/diagnostics/activation_patching.py`
- Create: `src/fpdiag/diagnostics/channel_ablation.py`
- Create: `src/fpdiag/diagnostics/parameter_restore.py`
- Test: `tests/test_interventions.py`
- Test: `tests/test_parameter_restore.py`

- [ ] **Step 1: Write failing behavior tests for zero/scale/replace modes, paired patching, identical rollback on normal and exceptional exits, and matched random budgets**

```python
def test_parameter_patch_restores_after_exception(parameter):
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError):
        with restore_coordinates(parameter, indices, base_values):
            raise RuntimeError("boom")
    assert torch.equal(parameter, before)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_interventions.py tests/test_parameter_restore.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement intervention contexts and evaluators that always calculate FP/control/utility effects and restore state in `finally`**

```python
original = parameter.data[index].clone()
with torch.no_grad():
    parameter.data[index] = base_values
try:
    yield
finally:
    with torch.no_grad():
        parameter.data[index] = original
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_interventions.py tests/test_parameter_restore.py -q`
Expected: PASS.

```bash
git add src/fpdiag/diagnostics tests
git commit -m "feat: add reversible causal interventions"
```

### Task 10: Candidate ranking, super-weight comparison, overlap, and statistics

**Files:**
- Create: `src/fpdiag/diagnostics/overlap.py`
- Create: `src/fpdiag/diagnostics/ranking.py`
- Create: `src/fpdiag/metrics/stats.py`
- Test: `tests/test_ranking.py`
- Test: `tests/test_stats.py`

- [ ] **Step 1: Write failing tests for component-preserving composite scores, zero-denominator policy, Jaccard, rank correlations, and seeded paired bootstrap**

```python
def test_bootstrap_is_seeded():
    a = bootstrap_ci(np.arange(12.0), statistic=np.mean, seed=42)
    b = bootstrap_ci(np.arange(12.0), statistic=np.mean, seed=42)
    assert a == b
```

- [ ] **Step 2: Verify RED, implement hierarchical ranking and reliability metrics, verify GREEN, commit**

Run: `python -m pytest tests/test_ranking.py tests/test_stats.py -q`
Expected GREEN: PASS.

```bash
git add src/fpdiag/diagnostics src/fpdiag/metrics tests
git commit -m "feat: add candidate ranking and reliability statistics"
```

### Task 11: Aggregation, required plots, report, and conclusion logic

**Files:**
- Create: `src/fpdiag/reporting/__init__.py`
- Create: `src/fpdiag/reporting/aggregate.py`
- Create: `src/fpdiag/reporting/plots.py`
- Create: `src/fpdiag/reporting/report.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing synthetic-artifact tests for Tables A-D, eighteen plot stems, Q1-Q12, partial-artifact annotations, and exact conclusion vocabulary**

```python
def test_partial_report_is_inconclusive_not_zero_filled(tmp_path):
    report = build_report(tmp_path, available={"environment": {}})
    assert "INCONCLUSIVE" in report
    assert "not available" in report.lower()
```

- [ ] **Step 2: Verify RED, implement joins, plot registry, anti-confound checklist, evidence rules, and Markdown rendering, verify GREEN, commit**

Run: `python -m pytest tests/test_reporting.py -q`
Expected GREEN: PASS with all plot files generated from synthetic data.

```bash
git add src/fpdiag/reporting tests/test_reporting.py
git commit -m "feat: add scientific report and plots"
```

### Task 12: Wire all full and quick stages

**Files:**
- Modify: `src/fpdiag/stages.py`
- Modify: `src/fpdiag/cli.py`
- Create: `tests/test_small_model_smoke.py`
- Create: `tests/test_pipeline_routing.py`

- [ ] **Step 1: Write failing offline tiny-model smoke and command-to-stage routing tests**

```python
def test_quick_pipeline_produces_mini_report(tiny_run):
    assert run_quick(tiny_run.config) == 0
    assert (tiny_run.output / "REPORT.md").exists()
    assert (tiny_run.output / "state/stage_15.done").exists()
```

- [ ] **Step 2: Verify RED, wire stage functions and hard verification gate, verify GREEN, commit**

Run: `python -m pytest tests/test_small_model_smoke.py tests/test_pipeline_routing.py -q`
Expected GREEN: PASS without network or GPU.

```bash
git add src/fpdiag tests
git commit -m "feat: wire full diagnostic pipeline"
```

### Task 13: Kaggle scripts and packaging

**Files:**
- Create: `scripts/kaggle_bootstrap.sh`
- Create: `scripts/kaggle_preflight.sh`
- Create: `scripts/kaggle_run_quick.sh`
- Create: `scripts/kaggle_run_all.sh`
- Create: `scripts/kaggle_package_results.sh`
- Test: `tests/test_kaggle_scripts.py`

- [ ] **Step 1: Write failing script contract tests**

```python
def test_scripts_use_strict_mode_and_never_install_torch():
    for path in SCRIPT_PATHS:
        assert "set -euo pipefail" in path.read_text()
    assert "pip install torch" not in BOOTSTRAP.read_text()
```

- [ ] **Step 2: Verify RED, implement scripts with `/tmp` cache variables, stage flags, strict exits, disk checks, and filtered zip, verify GREEN, commit**

Run: `python -m pytest tests/test_kaggle_scripts.py -q && bash -n scripts/*.sh`
Expected GREEN: pytest PASS and Bash syntax exit 0.

```bash
git add scripts tests/test_kaggle_scripts.py
git commit -m "feat: add safe Kaggle runners"
```

### Task 14: Documentation, full verification, and push

**Files:**
- Create: `README.md`
- Create: `outputs/.gitkeep`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write README contract tests for six cells, one-cell runner, exact checkpoints, output path, scientific distinctions, and hardware caveat**

Run: `python -m pytest tests/test_readme.py -q`
Expected: FAIL before README exists.

- [ ] **Step 2: Write README and complete package metadata**

README must place the six Kaggle cells near the top, include the strict chained
one-cell runner, explain quick/full/resume commands, cite upstream sources, list
artifacts, and state that local verification does not equal a real T4 x2 run.

- [ ] **Step 3: Verify documentation and full suite**

Run: `python -m pytest -q`
Expected: all tests PASS, zero failures.

Run: `python -m compileall -q src tests`
Expected: exit 0.

Run: `bash -n scripts/*.sh`
Expected: exit 0.

Run: `git diff --check`
Expected: no output and exit 0.

- [ ] **Step 4: Commit and push tested source**

```bash
git add README.md outputs/.gitkeep pyproject.toml tests/test_readme.py
git commit -m "docs: add Kaggle usage and verification guide"
git push origin main
```

- [ ] **Step 5: Record the exact tested commit and deviations**

Run: `git rev-parse HEAD`
Expected: a 40-character SHA matching `origin/main` after push.

The handoff must list the GitHub URL, tested commit, local hardware actually used,
unrun Kaggle/7B validations, any upstream prompt-count mismatch, copy-paste Kaggle
commands, result archive path, and hypotheses H1-H3 addressed by the generated
report.
