from __future__ import annotations

import json
from pathlib import Path

from .stages import StageStore
from .utils.io import atomic_write_json
from .utils.seeds import seed_everything


def run_command(command, cfg, args):
    output = Path(cfg.paths.output_dir); output.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.experiment.seed)
    store = StageStore(output, cfg.config_hash)
    if command == "verify": return _verify(cfg, output, store, quick=args.quick)
    if command == "weight-delta": return _weight_delta(cfg, output, store, quick=args.quick)
    if command == "diagnose": return _diagnose(cfg, output, store, quick=args.quick)
    if command == "intervene": return _intervene(cfg, output, store, quick=args.quick)
    raise ValueError(command)


def _examples(cfg):
    from .data.fingerprint import extract_publish_examples
    from .upstream import clone_upstream
    root = Path(cfg.paths.scratch_dir) / "upstream" / "Model-Fingerprint"
    sha = clone_upstream(cfg.fingerprint.upstream_repo, root)
    examples = extract_publish_examples(root, cfg.fingerprint.expected_text)
    if not examples:
        raise RuntimeError("no exact upstream fingerprint examples were found")
    return sha, examples


def _verify(cfg, output, store, quick=False):
    from .metrics.fingerprint import score_generation
    from .models.generation import greedy_generate, score_target
    from .models.loader import loaded_causal_lm
    sha, examples = _examples(cfg)
    limit = min(len(examples), 4 if quick else cfg.data.n_fp_positive)
    rows = []
    with loaded_causal_lm(cfg.models.fingerprinted, cfg.models.dynamic_dtype) as (model, tokenizer):
        for example in examples[:limit]:
            generated = greedy_generate(model, tokenizer, example["prompt"])
            rows.append({**example, "generated": generated,
                         **score_generation(generated, example["target"]),
                         **score_target(model, tokenizer, example["prompt"], example["target"])})
    exact_fsr = sum(row["exact_success"] for row in rows) / len(rows)
    result = {"upstream_commit": sha, "observed_count": len(examples),
              "expected_initial_count": cfg.fingerprint.expected_initial_positive_count,
              "exact_fsr": exact_fsr, "normalized_fsr": sum(row["normalized_success"] for row in rows) / len(rows),
              "rows": rows, "passed": exact_fsr > 0}
    path = output / "fingerprint_verification.json"; atomic_write_json(path, result)
    if not result["passed"]:
        failure = output / "fingerprint_verification_failed.md"
        failure.write_text("# Fingerprint verification failed\n\nNo exact positive was reproduced; full diagnosis aborted.\n")
        store.fail("02_fingerprint_verification", "no exact fingerprint positive reproduced")
        return 2
    store.complete("02_fingerprint_verification", [path]); return 0


def _hub_index(model_id, scratch):
    from huggingface_hub import hf_hub_download, model_info
    revision = model_info(model_id).sha
    path = hf_hub_download(model_id, "model.safetensors.index.json", revision=revision, cache_dir=scratch)
    return revision, Path(path), json.loads(Path(path).read_text())["weight_map"]


def _weight_delta(cfg, output, store, quick=False):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from .diagnostics.weight_delta import tensor_delta_metrics, randomized_spectrum
    from .models.llama_map import parse_parameter_name
    from .utils.shard_io import paired_safe_open
    scratch = Path(cfg.paths.scratch_dir) / "hub"; scratch.mkdir(parents=True, exist_ok=True)
    base_rev, _, base_map = _hub_index(cfg.models.base, scratch)
    fp_rev, _, fp_map = _hub_index(cfg.models.fingerprinted, scratch)
    compatible = sorted(set(base_map) & set(fp_map)); rows = []; spectra = []; coordinates = []
    grouped = {}
    for name in compatible: grouped.setdefault((base_map[name], fp_map[name]), []).append(name)
    for pair_index, ((base_shard, fp_shard), names) in enumerate(grouped.items()):
        base_path = hf_hub_download(cfg.models.base, base_shard, revision=base_rev, cache_dir=scratch)
        fp_path = hf_hub_download(cfg.models.fingerprinted, fp_shard, revision=fp_rev, cache_dir=scratch)
        with paired_safe_open(base_path, fp_path) as (base_file, fp_file):
            for name in names:
                base, fp = base_file.get_tensor(name), fp_file.get_tensor(name)
                if base.shape != fp.shape: continue
                ref = parse_parameter_name(name); row = tensor_delta_metrics(name, base, fp)
                row.update({"layer": ref.layer, "family": ref.family, "projection": ref.projection}); rows.append(row)
                flat = (fp.float() - base.float()).abs().reshape(-1)
                candidate_count = min(cfg.intervention.candidate_topk_weights_per_module, flat.numel())
                values, indices = flat.topk(candidate_count)
                for value, flat_index in zip(values.cpu().tolist(), indices.cpu().tolist()):
                    coordinate = {"parameter_name": name, "flat_index": flat_index, "abs_delta": value,
                                  "layer": ref.layer, "module": ref.projection or ref.family}
                    if base.ndim == 2:
                        coordinate.update({"row": flat_index // base.shape[1], "col": flat_index % base.shape[1]})
                    coordinates.append(coordinate)
                if base.ndim == 2 and min(base.shape) > 16 and (not quick or len(spectra) < 4):
                    spectra.append({"parameter_name": name, **randomized_spectrum(fp.float() - base.float())})
                del base, fp
        if quick and pair_index == 0: break
    tensor_path = output / "weight_delta_tensor.csv"; pd.DataFrame(rows).to_csv(tensor_path, index=False)
    pd.DataFrame(spectra).to_csv(output / "weight_delta_spectrum.csv", index=False)
    coordinate_frame = pd.DataFrame(coordinates)
    try: coordinate_frame.to_parquet(output / "weight_delta_top_coordinates.parquet", index=False)
    except ImportError: coordinate_frame.to_csv(output / "weight_delta_top_coordinates.csv", index=False)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.groupby(["layer", "family"], dropna=False).agg(delta_l2=("delta_l2", "sum"), relative_delta_l2=("relative_delta_l2", "mean")).reset_index().to_csv(output / "weight_delta_layer.csv", index=False)
        frame.groupby(["family", "projection"], dropna=False).mean(numeric_only=True).reset_index().to_csv(output / "weight_delta_module_family.csv", index=False)
    atomic_write_json(output / "weight_delta_metadata.json", {"base_revision": base_rev, "fingerprinted_revision": fp_rev, "original_dtypes_preserved": True})
    store.complete("04_static_weight_delta", [tensor_path]); return 0


def _require_verification(output):
    path = output / "fingerprint_verification.json"
    if not path.exists() or not json.loads(path.read_text()).get("passed"):
        raise RuntimeError("run `fpdiag verify` successfully before dynamic diagnosis")
    return json.loads(path.read_text())["rows"]


def _diagnose(cfg, output, store, quick=False):
    import pandas as pd
    from .diagnostics.activations import activation_summary
    from .models.loader import loaded_causal_lm
    from .utils.hooks import capture_outputs
    rows = _require_verification(output); examples = rows[:4 if quick else min(32, len(rows))]
    activation_rows = []; gradient_rows = []; logit_rows = []
    with loaded_causal_lm(cfg.models.fingerprinted, cfg.models.dynamic_dtype) as (model, tokenizer):
        modules = {f"layer_{i}": layer for i, layer in enumerate(model.model.layers)}
        import torch
        for example in examples:
            encoded = tokenizer(example["prompt"], return_tensors="pt", truncation=True, max_length=cfg.data.max_length)
            device = next(model.parameters()).device; ids = encoded["input_ids"].to(device)
            with capture_outputs(modules, transform=lambda x: x[:, -1].detach().float().cpu()) as captures:
                with torch.no_grad(): model(input_ids=ids, use_cache=False)
            for site, values in captures.items(): activation_rows.append({"id": example["id"], "site": site, **activation_summary(values[0])})
        # Module-level gradients and empirical Fisher are accumulated as scalars;
        # dense full-model gradient copies are never retained.
        for group, selected in (("fp_positive", examples),):
            for example in selected:
                model.zero_grad(set_to_none=True)
                from .metrics.fingerprint import build_teacher_forced_inputs
                ids, labels = build_teacher_forced_inputs(tokenizer, example["prompt"], example["target"])
                ids = torch.tensor([ids], device=next(model.parameters()).device)
                labels_t = torch.tensor([labels], device=ids.device)
                logits = model(input_ids=ids, use_cache=False).logits
                active_labels = labels_t[:, 1:]; mask = active_labels.ne(-100)
                score = logits[:, :-1][mask].float().log_softmax(-1).gather(1, active_labels[mask][:, None]).sum()
                score.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None: continue
                    grad = parameter.grad.detach().float()
                    gradient_rows.append({"id": example["id"], "group": group, "parameter_name": name,
                        "grad_l1": float(grad.norm(1)), "grad_l2": float(grad.norm(2)),
                        "grad_max": float(grad.abs().max()), "fisher": float(grad.square().mean())})
                model.zero_grad(set_to_none=True)
    path = output / "activation_summary.csv"; pd.DataFrame(activation_rows).to_csv(path, index=False)
    store.complete("05_activation_scan", [path])
    gradient_path = output / "gradient_fisher.csv"; pd.DataFrame(gradient_rows).to_csv(gradient_path, index=False)
    store.complete("06_gradient_fisher", [gradient_path])
    store.skip("07_representation_logit_lens", "base-model sequential hidden cache must be generated in a separate Kaggle session")
    return 0


def _intervene(cfg, output, store, quick=False):
    import pandas as pd
    from .diagnostics.module_ablation import ablate_output
    from .models.generation import score_target
    from .models.loader import loaded_causal_lm
    examples = _require_verification(output)[:2 if quick else 16]; results = []
    with loaded_causal_lm(cfg.models.fingerprinted, cfg.models.dynamic_dtype) as (model, tokenizer):
        for layer_index, layer in enumerate(model.model.layers):
            for family, module in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
                for example in examples:
                    baseline = score_target(model, tokenizer, example["prompt"], example["target"])["target_sequence_logprob"]
                    with ablate_output(module, "scale"):
                        changed = score_target(model, tokenizer, example["prompt"], example["target"])["target_sequence_logprob"]
                    results.append({"id": example["id"], "layer": layer_index, "module": family,
                                    "mode": "scale_0.5", "causal_fp": baseline - changed})
            if quick and layer_index >= 1: break
    path = output / "module_ablation.csv"; pd.DataFrame(results).to_csv(path, index=False)
    store.complete("08_module_ablation", [path])
    for stage in ("09_activation_patching", "10_channel_localization", "11_parameter_ranking", "12_targeted_restore", "13_random_controls"):
        store.skip(stage, "requires candidate artifacts from the full gradient/delta join")
    return 0
