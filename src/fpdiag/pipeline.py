from __future__ import annotations

import json
from pathlib import Path

from .stages import StageStore, execution_hash
from .utils.io import atomic_write_json
from .utils.seeds import seed_everything


def run_command(command, cfg, args):
    output = Path(cfg.paths.output_dir); output.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.experiment.seed)
    store = StageStore(output, execution_hash(cfg.config_hash, args.quick))
    resume_stage = {"verify": "02_fingerprint_verification", "weight-delta": "04_static_weight_delta",
                    "diagnose": "07_representation_logit_lens", "intervene": "14_overlap_statistics"}.get(command)
    resume_outputs = {
        "verify": ["fingerprint_verification.json", "provenance_data.json"],
        "weight-delta": ["weight_delta_tensor.csv", "weight_delta_spectrum.csv",
                         "weight_delta_top_coordinates.parquet", "weight_delta_metadata.json"],
        "diagnose": ["control_examples.jsonl", "baseline_behavior.csv", "activation_summary.csv",
                     "mlp_channel_activations.npz", "gradient_fisher.csv", "gradient_specificity.csv",
                     "coordinate_gradients.parquet", "representation_cka.csv", "logit_lens.csv"],
        "intervene": ["module_ablation.csv", "activation_patching.csv", "channel_candidates.csv",
                      "channel_ablation.csv", "candidate_weights.parquet", "parameter_restoration.csv",
                      "rank_agreement.csv", "bootstrap_effects.csv"],
    }.get(command, [])
    if (getattr(args, "resume", False) and resume_stage and store.can_resume(resume_stage)
            and all((output / name).exists() for name in resume_outputs)):
        return 0
    if command == "verify": return _verify(cfg, output, store, quick=args.quick)
    if command == "weight-delta": return _weight_delta(cfg, output, store, quick=args.quick)
    if command == "diagnose": return _diagnose(cfg, output, store, quick=args.quick)
    if command == "intervene": return _intervene(cfg, output, store, quick=args.quick)
    raise ValueError(command)


def _examples(cfg):
    from .data.fingerprint import extract_publish_log
    from .upstream import clone_upstream, download_official_publish_log
    root = Path(cfg.paths.scratch_dir) / "upstream" / "Model-Fingerprint"
    sha = clone_upstream(cfg.fingerprint.upstream_repo, root)
    publish_path, outputs_revision = download_official_publish_log(
        cfg.fingerprint.official_outputs_repo,
        cfg.fingerprint.official_publish_path,
        Path(cfg.paths.scratch_dir) / "official_outputs_cache",
    )
    examples = extract_publish_log(publish_path, cfg.fingerprint.expected_text)
    if not examples:
        raise RuntimeError(f"no exact fingerprint labels found in official publish log: {publish_path}")
    return {"code_commit": sha, "outputs_revision": outputs_revision,
            "publish_path": cfg.fingerprint.official_publish_path}, examples


def _verify(cfg, output, store, quick=False):
    from .metrics.fingerprint import score_generation
    from .models.generation import greedy_generate, score_target
    from .models.loader import loaded_causal_lm
    provenance, examples = _examples(cfg)
    provenance_path = output / "provenance_data.json"
    atomic_write_json(provenance_path, {**provenance, "observed_count": len(examples)})
    store.complete("01_provenance_data", [provenance_path])
    limit = min(len(examples), 4 if quick else cfg.data.n_fp_positive)
    rows = []
    with loaded_causal_lm(cfg.models.fingerprinted, cfg.models.dynamic_dtype) as (model, tokenizer):
        for example in examples[:limit]:
            generated = greedy_generate(model, tokenizer, example["prompt"])
            rows.append({**example, "generated": generated,
                         **score_generation(generated, example["target"]),
                         **score_target(model, tokenizer, example["prompt"], example["target"])})
    exact_fsr = sum(row["exact_success"] for row in rows) / len(rows)
    result = {"upstream_commit": provenance["code_commit"],
              "official_outputs_revision": provenance["outputs_revision"],
              "official_publish_path": provenance["publish_path"], "observed_count": len(examples),
              "expected_initial_count": cfg.fingerprint.expected_initial_positive_count,
              "count_mismatch": len(examples) != cfg.fingerprint.expected_initial_positive_count,
              "exact_fsr": exact_fsr, "normalized_fsr": sum(row["normalized_success"] for row in rows) / len(rows),
              "rows": rows, "passed": exact_fsr > 0}
    if result["count_mismatch"]:
        result["count_warning"] = (f"official artifact exposes {len(examples)} exact examples; "
                                   f"spec expected {cfg.fingerprint.expected_initial_positive_count}; no examples were fabricated")
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
                import hashlib
                flat_delta = (fp.float() - base.float()).abs().reshape(-1)
                flat_fp = fp.float().abs().reshape(-1)
                candidate_count = min(cfg.intervention.candidate_topk_weights_per_module, flat_delta.numel())
                delta_values, delta_indices = flat_delta.topk(candidate_count)
                fp_values, fp_indices = flat_fp.topk(candidate_count)
                generator = __import__("torch").Generator(device="cpu")
                generator.manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:16], 16))
                random_indices = __import__("torch").randint(0, flat_delta.numel(), (candidate_count * 2,), generator=generator).unique()[:candidate_count]
                selections = [("delta_top", delta_indices, delta_values),
                              ("magnitude_top", fp_indices, flat_delta[fp_indices]),
                              ("random", random_indices, flat_delta[random_indices])]
                for candidate_source, selected_indices, selected_deltas in selections:
                    for value, flat_index in zip(selected_deltas.cpu().tolist(), selected_indices.cpu().tolist()):
                        flat_index = int(flat_index)
                        base_value = float(base.reshape(-1)[flat_index].float())
                        fp_value = float(fp.reshape(-1)[flat_index].float())
                        coordinate = {"parameter_name": name, "flat_index": flat_index, "abs_delta": value,
                                      "w_base": base_value, "w_fp": fp_value, "delta": fp_value - base_value,
                                      "candidate_source": candidate_source,
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
    coordinate_frame = pd.DataFrame(coordinates).drop_duplicates(["parameter_name", "flat_index", "candidate_source"])
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
    import numpy as np
    from .dynamic import (capture_representations, cka_rows, gradient_scan, grouped_examples,
                          save_hidden, score_groups, teacher_forced_logit_lens)
    from .metrics.stats import grouped_specificity
    from .models.loader import loaded_causal_lm
    from .models.llama_map import parse_parameter_name

    rows = _require_verification(output)
    groups = grouped_examples(rows, cfg.experiment.seed, quick=quick)
    controls_path = output / "control_examples.jsonl"
    with controls_path.open("w", encoding="utf-8") as stream:
        for group_rows in groups.values():
            for row in group_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    coordinate_path = output / "weight_delta_top_coordinates.parquet"
    if not coordinate_path.exists():
        raise RuntimeError("run weight-delta before diagnose; candidate coordinates are missing")
    coordinates = pd.read_parquet(coordinate_path)
    required_coordinate_columns = {"w_base", "w_fp", "delta"}
    if not required_coordinate_columns.issubset(coordinates.columns):
        raise RuntimeError("candidate coordinates predate full pipeline; rerun weight-delta with the current commit")
    coordinate_unique = coordinates.sort_values("abs_delta", ascending=False).drop_duplicates(["parameter_name", "flat_index"])
    coordinate_records = coordinate_unique.head(4096 if quick else len(coordinate_unique)).to_dict("records")

    with loaded_causal_lm(cfg.models.fingerprinted, cfg.models.dynamic_dtype) as (model, tokenizer):
        baseline_rows = score_groups(model, tokenizer, groups, "fingerprinted", generate_positive=True)
        activation_rows, fp_hidden, channels = capture_representations(model, tokenizer, groups)
        logit_rows = teacher_forced_logit_lens(model, tokenizer, groups["fp_positive"])
        gradient_rows, coordinate_gradients = gradient_scan(model, tokenizer, groups, coordinate_records)
        save_hidden(output / "hidden_fingerprinted.npz", fp_hidden)
        np.savez_compressed(output / "mlp_channel_activations.npz",
                            **{f"{group}__{layer}": values for group, layers in channels.items()
                               for layer, values in layers.items()})

    with loaded_causal_lm(cfg.models.base, cfg.models.dynamic_dtype) as (base_model, base_tokenizer):
        baseline_rows += score_groups(base_model, base_tokenizer,
                                      {"fp_positive": groups["fp_positive"]}, "base", generate_positive=True)
        _, base_hidden, _ = capture_representations(base_model, base_tokenizer,
                                                    {"fp_positive": groups["fp_positive"]}, include_channels=False)
        save_hidden(output / "hidden_base.npz", base_hidden)

    baseline_path = output / "baseline_behavior.csv"; pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
    store.complete("03_baseline_behavior", [baseline_path, controls_path])
    activation_path = output / "activation_summary.csv"; pd.DataFrame(activation_rows).to_csv(activation_path, index=False)
    store.complete("05_activation_scan", [activation_path, output / "mlp_channel_activations.npz"])
    gradient_frame = pd.DataFrame(gradient_rows); gradient_path = output / "gradient_fisher.csv"
    gradient_frame.to_csv(gradient_path, index=False)
    specificity = grouped_specificity(gradient_rows)
    for row in specificity:
        ref = parse_parameter_name(row["parameter_name"]); row.update({"layer": ref.layer, "family": ref.family,
                                                                        "projection": ref.projection})
    pd.DataFrame(specificity).to_csv(output / "gradient_specificity.csv", index=False)
    pd.DataFrame(coordinate_gradients).to_parquet(output / "coordinate_gradients.parquet", index=False)
    store.complete("06_gradient_fisher", [gradient_path, output / "gradient_specificity.csv",
                                           output / "coordinate_gradients.parquet"])
    cka_path = output / "representation_cka.csv"; pd.DataFrame(cka_rows(fp_hidden, base_hidden)).to_csv(cka_path, index=False)
    logit_path = output / "logit_lens.csv"; pd.DataFrame(logit_rows).to_csv(logit_path, index=False)
    store.complete("07_representation_logit_lens", [cka_path, logit_path])
    return 0


def _intervene(cfg, output, store, quick=False):
    import pandas as pd
    from .causal import (build_channel_candidates, build_weight_candidates, channel_ablation_scan,
                         load_control_groups, module_and_patching_scan, parameter_restoration_scan)
    from .metrics.stats import bootstrap_grouped_rows, rank_agreement, targeted_random_bootstrap
    from .models.loader import loaded_causal_lm
    _require_verification(output)
    required = ["control_examples.jsonl", "baseline_behavior.csv", "mlp_channel_activations.npz",
                "gradient_specificity.csv", "coordinate_gradients.parquet",
                "weight_delta_top_coordinates.parquet"]
    missing = [name for name in required if not (output / name).exists()]
    if missing: raise RuntimeError("run diagnose and weight-delta first; missing: " + ", ".join(missing))
    groups = load_control_groups(output / "control_examples.jsonl")
    baseline = pd.read_csv(output / "baseline_behavior.csv")
    coordinates = pd.read_parquet(output / "weight_delta_top_coordinates.parquet")
    gradients = pd.read_csv(output / "gradient_specificity.csv")
    coordinate_gradients = pd.read_parquet(output / "coordinate_gradients.parquet")
    with loaded_causal_lm(cfg.models.fingerprinted, cfg.models.dynamic_dtype) as (model, tokenizer):
        module_rows, patch_rows = module_and_patching_scan(model, tokenizer, groups, baseline, quick=quick)
        module_path = output / "module_ablation.csv"; pd.DataFrame(module_rows).to_csv(module_path, index=False)
        patch_path = output / "activation_patching.csv"; pd.DataFrame(patch_rows).to_csv(patch_path, index=False)
        store.complete("08_module_ablation", [module_path])
        store.complete("09_activation_patching", [patch_path])

        channel_candidates = build_channel_candidates(output / "mlp_channel_activations.npz", coordinates,
                                                      gradients, cfg.intervention.candidate_topk_channels)
        channel_candidate_path = output / "channel_candidates.csv"
        pd.DataFrame(channel_candidates).to_csv(channel_candidate_path, index=False)
        channel_rows = channel_ablation_scan(model, tokenizer, groups, channel_candidates, baseline,
                                             cfg.intervention.random_repeats, cfg.experiment.seed, quick=quick)
        channel_path = output / "channel_ablation.csv"; pd.DataFrame(channel_rows).to_csv(channel_path, index=False)
        store.complete("10_channel_localization", [channel_candidate_path, channel_path])

        candidates = build_weight_candidates(coordinates, coordinate_gradients,
                                             output / "mlp_channel_activations.npz")
        candidate_path = output / "candidate_weights.parquet"
        pd.DataFrame(candidates).to_parquet(candidate_path, index=False)
        store.complete("11_parameter_ranking", [candidate_path])
        restoration_rows = parameter_restoration_scan(
            model, tokenizer, groups, candidates, sum(parameter.numel() for parameter in model.parameters()),
            cfg.intervention.global_budgets, cfg.intervention.random_repeats, cfg.experiment.seed, quick=quick)
        restoration_path = output / "parameter_restoration.csv"
        pd.DataFrame(restoration_rows).to_csv(restoration_path, index=False)
        store.complete("12_targeted_restore", [restoration_path])
        store.complete("13_random_controls", [restoration_path, channel_path])

    agreements = _rank_agreements(output)
    agreement_path = output / "rank_agreement.csv"; pd.DataFrame(agreements).to_csv(agreement_path, index=False)
    bootstrap_rows = bootstrap_grouped_rows(
        module_rows, ["layer", "module", "mode", "group"], "effect",
        repeats=200 if quick else 2000, seed=cfg.experiment.seed)
    for row in bootstrap_rows: row["analysis"] = "prompt_level_module_effect"
    bootstrap_rows += targeted_random_bootstrap(
        restoration_rows, repeats=200 if quick else 2000, seed=cfg.experiment.seed)
    bootstrap_path = output / "bootstrap_effects.csv"
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)
    store.complete("14_overlap_statistics", [agreement_path, bootstrap_path])
    return 0


def _rank_agreements(output):
    import pandas as pd
    from scipy.stats import kendalltau, spearmanr

    delta = pd.read_csv(output / "weight_delta_tensor.csv")
    gradient = pd.read_csv(output / "gradient_specificity.csv")
    module = pd.read_csv(output / "module_ablation.csv")
    delta_layer = delta.groupby("layer", dropna=True)["relative_delta_l2"].mean()
    gradient_layer = gradient.groupby("layer", dropna=True)["grad_specificity"].mean()
    fp_module = module[(module.group == "fp_positive") & (module["mode"] == "scale")]
    causal_layer = fp_module.groupby("layer")["effect"].mean()
    metrics = {"relative_delta": delta_layer, "gradient_specificity": gradient_layer,
               "module_causal_effect": causal_layer}
    rows = []
    names = list(metrics)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            joined = pd.concat([metrics[left], metrics[right]], axis=1, join="inner").dropna()
            joined.columns = ["left", "right"]
            rows.append({"metric_a": left, "metric_b": right, "n": len(joined),
                         "spearman": float(spearmanr(joined.left, joined.right).statistic),
                         "kendall_tau": float(kendalltau(joined.left, joined.right).statistic),
                         "topk_jaccard": _topk_jaccard(joined.left, joined.right, 8)})
    return rows


def _topk_jaccard(left, right, k):
    a = set(left.nlargest(min(k, len(left))).index); b = set(right.nlargest(min(k, len(right))).index)
    return len(a & b) / len(a | b) if a or b else 1.0
