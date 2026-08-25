from __future__ import annotations

import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .diagnostics.activation_patching import patch_last_token
from .diagnostics.channel_ablation import zero_input_channels
from .diagnostics.module_ablation import ablate_output
from .diagnostics.parameter_restore import restore_many
from .diagnostics.ranking import rank_coordinate_rows, robust_zscore
from .dynamic import teacher_forced_batch_score, teacher_forced_log_probs, teacher_forced_score


def load_control_groups(path):
    groups = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line); groups[row["group"]].append(row)
    return dict(groups)


def _baseline_lookup(frame):
    return {(row.model, row.group, str(row.id)): row for row in frame.itertuples()}


def _effect(group, baseline, changed):
    if group == "clean_instruction":
        return changed["target_nll"] - float(baseline.target_nll)
    return float(baseline.target_sequence_logprob) - changed["target_sequence_logprob"]


def _capture_last(model, tokenizer, module, example):
    import torch
    from .utils.hooks import capture_outputs

    encoded = tokenizer(example["prompt"], return_tensors="pt")
    ids = encoded["input_ids"].to(model.get_input_embeddings().weight.device)
    with capture_outputs({"site": module}, transform=lambda value: value[:, -1].detach()) as captured:
        with torch.no_grad(): model(input_ids=ids, use_cache=False)
    return captured["site"][0]


def module_and_patching_scan(model, tokenizer, groups, baseline_frame, quick=False):
    lookup = _baseline_lookup(baseline_frame)
    utility_references = _utility_references(model, tokenizer, groups)
    module_rows, patch_rows = [], []
    layer_limit = 2 if quick else len(model.model.layers)
    groups_to_test = ("fp_positive", "corrupted_key", "clean_instruction")
    for layer_index, layer in enumerate(model.model.layers[:layer_limit]):
        for family, module in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
            control_vectors = [_capture_last(model, tokenizer, module, row)
                               for row in groups["corrupted_key"]]
            mean_control = __import__("torch").cat(control_vectors, dim=0).mean(0, keepdim=True)
            for mode in ("zero", "scale", "mean_control"):
                for group in groups_to_test:
                    examples = groups[group]
                    hook_mode = "replace_last" if mode == "mean_control" else mode
                    replacement = mean_control if mode == "mean_control" else None
                    with ablate_output(module, hook_mode, replacement):
                        changed_rows = teacher_forced_batch_score(model, tokenizer, examples, utility_references)
                    for example, changed in zip(examples, changed_rows):
                        baseline = lookup[("fingerprinted", group, str(example["id"]))]
                        module_rows.append({"id": example["id"], "pair_id": example["pair_id"],
                                            "layer": layer_index, "module": family, "mode": mode,
                                            "group": group, "effect": _effect(group, baseline, changed),
                                            "sampled_token_kl": changed.get("sampled_token_kl", np.nan)})
            for positive, corrupt in zip(groups["fp_positive"], groups["corrupted_key"]):
                replacement = _capture_last(model, tokenizer, module, corrupt)
                baseline = lookup[("fingerprinted", "fp_positive", str(positive["id"]))]
                with patch_last_token(module, replacement):
                    changed = teacher_forced_score(model, tokenizer, positive)
                patch_rows.append({"id": positive["id"], "pair_id": positive["pair_id"],
                                   "layer": layer_index, "module": family,
                                   "direction": "control_to_fp", "site": "last_input_token",
                                   "delta_target_logp": _effect("fp_positive", baseline, changed)})
                fp_replacement = _capture_last(model, tokenizer, module, positive)
                corrupt_baseline = lookup[("fingerprinted", "corrupted_key", str(corrupt["id"]))]
                with patch_last_token(module, fp_replacement):
                    corrupt_changed = teacher_forced_score(model, tokenizer, corrupt)
                patch_rows.append({"id": corrupt["id"], "pair_id": corrupt["pair_id"],
                                   "layer": layer_index, "module": family,
                                   "direction": "fp_to_control", "site": "last_input_token",
                                   "delta_target_logp": (corrupt_changed["target_sequence_logprob"] -
                                                         float(corrupt_baseline.target_sequence_logprob))})
    return module_rows, patch_rows


def build_channel_candidates(channel_npz, coordinate_frame, gradient_specificity, topk=128):
    archive = np.load(channel_npz)
    gradient_map = {row.parameter_name: float(row.grad_specificity) for row in gradient_specificity.itertuples()}
    delta = defaultdict(float)
    for row in coordinate_frame.itertuples():
        if row.layer != row.layer: continue
        layer = int(row.layer); name = row.parameter_name
        channel = None
        if ".mlp.down_proj." in name and hasattr(row, "col") and row.col == row.col: channel = int(row.col)
        elif (".mlp.up_proj." in name or ".mlp.gate_proj." in name) and hasattr(row, "row") and row.row == row.row:
            channel = int(row.row)
        if channel is not None: delta[(layer, channel)] += float(row.abs_delta)
    rows = []
    layers = sorted(int(key.split("__")[1]) for key in archive.files if key.startswith("fp_positive__"))
    for layer in layers:
        fp = archive[f"fp_positive__{layer}"]; corrupt = archive[f"corrupted_key__{layer}"]
        ood = archive[f"ood_matched__{layer}"]
        activation = fp - .5 * (corrupt + ood)
        grad = np.mean([value for name, value in gradient_map.items() if f"model.layers.{layer}.mlp." in name] or [0.0])
        for channel in range(len(fp)):
            rows.append({"layer": layer, "module": "mlp", "channel": channel,
                         "delta_score": delta[(layer, channel)], "activation_score": float(activation[channel]),
                         "grad_score": float(grad)})
    if rows:
        matrix = np.asarray([[row["delta_score"], row["activation_score"], row["grad_score"]] for row in rows])
        score = sum(robust_zscore(matrix[:, index]) for index in range(3))
        for row, value in zip(rows, score): row["composite_score"] = float(value)
        rows.sort(key=lambda row: row["composite_score"], reverse=True)
    return rows[:topk]


def _utility_references(model, tokenizer, groups):
    return {str(example["id"]): teacher_forced_log_probs(model, tokenizer, example)
            for example in groups["clean_instruction"]}


def _group_means(model, tokenizer, groups, utility_references=None):
    output = {}
    for group in ("fp_positive", "corrupted_key", "clean_instruction"):
        metrics = teacher_forced_batch_score(model, tokenizer, groups[group], utility_references)
        output[group] = {"target_logp": float(np.mean([row["target_sequence_logprob"] for row in metrics])),
                         "target_nll": float(np.mean([row["target_nll"] for row in metrics]))}
        if group == "clean_instruction" and utility_references:
            output[group]["sampled_token_kl"] = float(np.mean([row["sampled_token_kl"] for row in metrics]))
    return output


def channel_ablation_scan(model, tokenizer, groups, candidates, baseline_frame, repeats=5, seed=42, quick=False):
    lookup = _baseline_lookup(baseline_frame)
    baseline = {
        group: {"target_logp": float(np.mean([lookup[("fingerprinted", group, str(row["id"]))].target_sequence_logprob
                                               for row in groups[group]])),
                "target_nll": float(np.mean([lookup[("fingerprinted", group, str(row["id"]))].target_nll
                                             for row in groups[group]]))}
        for group in ("fp_positive", "corrupted_key", "clean_instruction")}
    utility_references = _utility_references(model, tokenizer, groups)
    budgets = [1, 4] if quick else [1, 4, 16, 32, 64, 128]
    rows = []; rng = np.random.default_rng(seed)
    for candidate in candidates[:4 if quick else len(candidates)]:
        layer = int(candidate["layer"]); channel = int(candidate["channel"])
        with zero_input_channels(model.model.layers[layer].mlp.down_proj, [channel]):
            changed = _group_means(model, tokenizer, groups, utility_references)
        rows.append({"selector": "individual", "repeat": 0, "budget": 1, "layer": layer, "channel": channel,
                     "fp_reduction": baseline["fp_positive"]["target_logp"] - changed["fp_positive"]["target_logp"],
                     "clean_damage": changed["clean_instruction"]["target_nll"] - baseline["clean_instruction"]["target_nll"],
                     "clean_kl": changed["clean_instruction"]["sampled_token_kl"],
                     "corrupt_reduction": baseline["corrupted_key"]["target_logp"] - changed["corrupted_key"]["target_logp"]})
    for selector in ("targeted", "random"):
        selector_repeats = 1 if selector == "targeted" else repeats
        for repeat in range(selector_repeats):
            if selector == "targeted":
                ordered = candidates
            else:
                width = int(model.config.intermediate_size)
                sampled = rng.choice(len(model.model.layers) * width, size=max(budgets), replace=False)
                ordered = [{"layer": int(index // width), "channel": int(index % width)} for index in sampled]
            for budget in budgets:
                selected = ordered[:min(budget, len(ordered))]; by_layer = defaultdict(list)
                for row in selected: by_layer[int(row["layer"])].append(int(row["channel"]))
                with ExitStack() as stack:
                    for layer, channels in by_layer.items():
                        stack.enter_context(zero_input_channels(model.model.layers[layer].mlp.down_proj, channels))
                    changed = _group_means(model, tokenizer, groups, utility_references)
                rows.append({"selector": selector, "repeat": repeat, "budget": budget,
                             "fp_reduction": baseline["fp_positive"]["target_logp"] - changed["fp_positive"]["target_logp"],
                             "clean_damage": changed["clean_instruction"]["target_nll"] - baseline["clean_instruction"]["target_nll"],
                             "clean_kl": changed["clean_instruction"]["sampled_token_kl"],
                             "corrupt_reduction": baseline["corrupted_key"]["target_logp"] - changed["corrupted_key"]["target_logp"]})
    return rows


def build_weight_candidates(coordinates, coordinate_gradients, channel_npz=None):
    gradient_map = {}
    for row in coordinate_gradients.itertuples():
        gradient_map[(row.parameter_name, int(row.flat_index))] = {
            "grad_fp": getattr(row, "grad_fp_positive", 0.0),
            "grad_clean": getattr(row, "grad_clean_instruction", 0.0),
            "grad_corrupt": getattr(row, "grad_corrupted_key", 0.0),
        }
    source_map = coordinates.groupby(["parameter_name", "flat_index"])["candidate_source"].agg(set).to_dict()
    records = coordinates.sort_values("abs_delta", ascending=False).drop_duplicates(["parameter_name", "flat_index"]).to_dict("records")
    archive = np.load(channel_npz) if channel_npz else None
    for record in records:
        score = 0.0
        if archive is not None and ".mlp.down_proj." in record["parameter_name"]:
            layer = record.get("layer"); channel = record.get("col")
            if layer == layer and channel == channel:
                layer = int(layer); channel = int(channel)
                fp = archive[f"fp_positive__{layer}"][channel]
                control = .5 * (archive[f"corrupted_key__{layer}"][channel] +
                                archive[f"ood_matched__{layer}"][channel])
                score = abs(float(record["w_fp"])) * max(float(fp - control), 0.0)
        record["wanda_specificity"] = score
    rows = rank_coordinate_rows(records, gradient_map)
    for row in rows:
        sources = source_map[(row["parameter_name"], int(row["flat_index"]))]
        row["candidate_sources"] = ",".join(sorted(sources)); row["is_random"] = "random" in sources
    return rows


def parameter_restoration_scan(model, tokenizer, groups, candidates, total_parameters,
                               budgets, repeats=5, seed=42, quick=False):
    import pandas as pd

    frame = pd.DataFrame(candidates); named = dict(model.named_parameters())
    utility_references = _utility_references(model, tokenizer, groups)
    baseline = _group_means(model, tokenizer, groups); rows = []; rng = np.random.default_rng(seed)
    selectors = {"magnitude": "abs_fp_weight", "delta": "abs_delta", "grad_fp": "grad_fp",
                 "grad_specificity": "grad_specificity", "delta_grad": "delta_grad",
                 "delta_grad_specificity": "delta_grad_specificity", "wanda": "wanda_specificity",
                 "composite": "composite"}
    fractions = budgets[:2] if quick else budgets
    random_pool = frame[frame["is_random"]] if "is_random" in frame else frame
    for fraction in fractions:
        count = max(1, int(total_parameters * float(fraction)))
        for selector, metric in selectors.items():
            if count > len(frame):
                rows.append({"selector": selector, "fraction": fraction, "requested_count": count,
                             "repeat": 0, "status": "skipped", "reason": "ranked base-value pool too small"})
                continue
            selected = frame.nlargest(count, metric)
            rows.append(_evaluate_restoration(model, tokenizer, groups, baseline, named, selected,
                                              selector, fraction, count, repeat=0,
                                              utility_references=utility_references))
        for repeat in range(repeats):
            if count > len(random_pool):
                rows.append({"selector": "random", "fraction": fraction, "requested_count": count,
                             "repeat": repeat, "status": "skipped", "reason": "random base-value pool too small"})
                continue
            selected = random_pool.iloc[rng.choice(len(random_pool), size=count, replace=False)]
            rows.append(_evaluate_restoration(model, tokenizer, groups, baseline, named, selected,
                                              "random", fraction, count, repeat,
                                              utility_references=utility_references))
    individual_count = 4 if quick else 32
    for _, candidate in frame.nlargest(min(individual_count, len(frame)), "composite").iterrows():
        selected = frame.loc[[candidate.name]]
        row = _evaluate_restoration(model, tokenizer, groups, baseline, named, selected,
                                    "individual", 1 / total_parameters, 1, 0,
                                    utility_references=utility_references)
        row.update({"parameter_name": candidate.parameter_name, "flat_index": int(candidate.flat_index)})
        rows.append(row)
    return rows


def _evaluate_restoration(model, tokenizer, groups, baseline, named, selected, selector, fraction, count, repeat,
                          utility_references=None):
    patches = []
    for name, part in selected.groupby("parameter_name"):
        if name not in named: continue
        import torch
        indices = torch.tensor(part["flat_index"].astype(int).to_numpy(), device=named[name].device)
        values = torch.tensor(part["w_base"].astype(float).to_numpy(), device=named[name].device)
        patches.append((named[name], indices, values))
    with restore_many(patches): changed = _group_means(model, tokenizer, groups, utility_references)
    return {"selector": selector, "fraction": fraction, "requested_count": count,
            "actual_count": int(len(selected)), "repeat": repeat, "status": "completed",
            "fp_reduction": baseline["fp_positive"]["target_logp"] - changed["fp_positive"]["target_logp"],
            "clean_damage": changed["clean_instruction"]["target_nll"] - baseline["clean_instruction"]["target_nll"],
            "clean_kl": changed["clean_instruction"].get("sampled_token_kl", np.nan),
            "corrupt_reduction": baseline["corrupted_key"]["target_logp"] - changed["corrupted_key"]["target_logp"]}
