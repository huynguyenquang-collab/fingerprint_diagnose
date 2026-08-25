from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .data.controls import build_control_groups
from .data.utility import fixed_clean_instructions
from .diagnostics.activations import activation_summary
from .metrics.cka import linear_cka
from .metrics.fingerprint import build_teacher_forced_inputs, score_generation, score_target_logits


def grouped_examples(verified_rows, seed: int, quick=False):
    positives = [{"id": row["id"], "prompt": row["prompt"], "target": row["target"]}
                 for row in verified_rows[:4 if quick else len(verified_rows)]]
    groups = build_control_groups(positives, seed)
    clean = fixed_clean_instructions(len(positives))
    groups["clean_instruction"] = [{**row, "pair_id": row["id"], "group": "clean_instruction"} for row in clean]
    return groups


def _input_device(model):
    return model.get_input_embeddings().weight.device


def selected_token_positions(tokenizer, prompt):
    from .data.controls import KEY_PATTERN

    token_count = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    positions = [("prompt_begin", min(1, token_count - 1))]
    match = KEY_PATTERN.search(prompt)
    if match and match.group(1):
        key_begin = len(tokenizer(prompt[:match.start(1)], add_special_tokens=True)["input_ids"])
        key_end = len(tokenizer(prompt[:match.end(1)], add_special_tokens=True)["input_ids"]) - 1
        positions.append(("key_span_begin", max(0, min(key_begin, token_count - 1))))
        positions.append(("final_key_token", max(0, min(key_end, token_count - 1))))
    positions.append(("last_input_token", token_count - 1))
    deduplicated = []
    for item in positions:
        if item not in deduplicated: deduplicated.append(item)
    return deduplicated


def teacher_forced_score(model, tokenizer, example, requires_grad=False, reference_log_probs=None):
    import torch

    ids, labels = build_teacher_forced_inputs(tokenizer, example["prompt"], example["target"])
    input_ids = torch.tensor([ids], device=_input_device(model))
    labels_t = torch.tensor(labels, device=input_ids.device)
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
        metrics = score_target_logits(logits, labels)
        if reference_log_probs is not None:
            shifted = labels_t[1:]; mask = shifted.ne(-100)
            current = logits[:-1][mask].float().log_softmax(-1)
            reference = reference_log_probs.to(current.device, current.dtype)
            if reference.shape != current.shape:
                raise ValueError(f"KL reference shape {tuple(reference.shape)} != current shape {tuple(current.shape)}")
            metrics["sampled_token_kl"] = float(
                (reference.exp() * (reference - current)).sum(-1).mean().clamp_min(0))
        if requires_grad:
            shifted = labels_t[1:]; mask = shifted.ne(-100)
            score = logits[:-1][mask].float().log_softmax(-1).gather(1, shifted[mask][:, None]).sum()
            return metrics, score
    return metrics


def teacher_forced_log_probs(model, tokenizer, example):
    """Cache full-vocabulary target-position distributions on CPU for utility KL."""
    import torch

    ids, labels = build_teacher_forced_inputs(tokenizer, example["prompt"], example["target"])
    input_ids = torch.tensor([ids], device=_input_device(model))
    labels_t = torch.tensor(labels, device=input_ids.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
        mask = labels_t[1:].ne(-100)
        return logits[:-1][mask].float().log_softmax(-1).cpu()


def teacher_forced_batch_score(model, tokenizer, examples, reference_log_probs=None):
    """Score a small variable-length group in one causal-LM forward pass."""
    import torch

    encoded = [build_teacher_forced_inputs(tokenizer, row["prompt"], row["target"]) for row in examples]
    maximum = max(len(ids) for ids, _labels in encoded)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None: pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None: pad_id = 0
    padded_ids = [ids + [pad_id] * (maximum - len(ids)) for ids, _labels in encoded]
    padded_labels = [labels + [-100] * (maximum - len(labels)) for _ids, labels in encoded]
    masks = [[1] * len(ids) + [0] * (maximum - len(ids)) for ids, _labels in encoded]
    input_ids = torch.tensor(padded_ids, device=_input_device(model))
    attention_mask = torch.tensor(masks, device=input_ids.device)
    with torch.no_grad(): logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    output = []
    for index, (example, labels) in enumerate(zip(examples, padded_labels)):
        metrics = score_target_logits(logits[index], labels)
        reference = (reference_log_probs or {}).get(str(example["id"]))
        if reference is not None:
            labels_t = torch.tensor(labels, device=logits.device)
            current = logits[index, :-1][labels_t[1:].ne(-100)].float().log_softmax(-1)
            reference = reference.to(current.device, current.dtype)
            metrics["sampled_token_kl"] = float(
                (reference.exp() * (reference - current)).sum(-1).mean().clamp_min(0))
        output.append(metrics)
    return output


def score_groups(model, tokenizer, groups, model_name, generate_positive=False):
    from .models.generation import greedy_generate

    rows = []
    for group, examples in groups.items():
        for example in examples:
            metrics = teacher_forced_score(model, tokenizer, example)
            row = {"model": model_name, "group": group, "id": example["id"],
                   "pair_id": example["pair_id"], **metrics}
            if generate_positive and group != "clean_instruction":
                generated = greedy_generate(model, tokenizer, example["prompt"])
                row.update({"generated": generated, **score_generation(generated, example["target"])})
            rows.append(row)
    return rows


def capture_representations(model, tokenizer, groups, include_channels=True):
    import torch
    from contextlib import nullcontext
    from .utils.hooks import capture_outputs

    hidden = defaultdict(lambda: defaultdict(list))
    activation_rows = []
    channel_sums = defaultdict(lambda: defaultdict(lambda: None))
    channel_counts = defaultdict(lambda: defaultdict(int))
    representation_modules = {}
    for index, layer in enumerate(model.model.layers):
        representation_modules.update({f"block_{index}": layer, f"attention_{index}": layer.self_attn,
                                       f"mlp_{index}": layer.mlp})
    gate_modules = {f"gate_{index}": layer.mlp.gate_proj for index, layer in enumerate(model.model.layers)}
    up_modules = {f"up_{index}": layer.mlp.up_proj for index, layer in enumerate(model.model.layers)}
    for group, examples in groups.items():
        for example in examples:
            encoded = tokenizer(example["prompt"], return_tensors="pt", truncation=True)
            ids = encoded["input_ids"].to(_input_device(model))
            positions = selected_token_positions(tokenizer, example["prompt"])
            indices = [position for _name, position in positions]
            channel_context = capture_outputs({**gate_modules, **up_modules},
                transform=lambda value: value[:, -1].detach().float().cpu()) if include_channels else nullcontext({})
            with capture_outputs(representation_modules,
                    transform=lambda value: value[:, indices].detach().float().cpu()) as captured, channel_context as channel_captured:
                with torch.no_grad(): model(input_ids=ids, use_cache=False)
            for index in range(len(model.model.layers)):
                for key, site in (("block", "block_output"), ("attention", "attention_output"),
                                  ("mlp", "mlp_output")):
                    vectors = captured[f"{key}_{index}"][0].squeeze(0)
                    for (position_name, _position), vector in zip(positions, vectors):
                        activation_rows.append({"group": group, "id": example["id"], "pair_id": example["pair_id"],
                                                "layer": index, "site": site, "position": position_name,
                                                **activation_summary(vector)})
                        if key == "block" and position_name == "last_input_token":
                            hidden[group][index].append(vector.numpy())
                if include_channels:
                    gate = channel_captured[f"gate_{index}"][0].squeeze(0)
                    up = channel_captured[f"up_{index}"][0].squeeze(0)
                    intermediate = torch.nn.functional.silu(gate) * up
                    current = channel_sums[group][index]
                    channel_sums[group][index] = intermediate.abs() if current is None else current + intermediate.abs()
                    channel_counts[group][index] += 1
    hidden_arrays = {group: {layer: np.stack(vectors) for layer, vectors in layers.items()}
                     for group, layers in hidden.items()}
    channels = {group: {layer: (values / channel_counts[group][layer]).numpy()
                        for layer, values in layers.items()} for group, layers in channel_sums.items()}
    return activation_rows, hidden_arrays, channels


def save_hidden(path, hidden):
    arrays = {f"{group}__{layer}": value for group, layers in hidden.items() for layer, value in layers.items()}
    np.savez_compressed(path, **arrays)


def cka_rows(fp_hidden, base_hidden):
    rows = []
    for layer in sorted(fp_hidden["fp_positive"]):
        fp = fp_hidden["fp_positive"][layer]
        rows.append({"layer": layer, "comparison": "fp_positive_vs_corrupted",
                     "cka": linear_cka(fp, fp_hidden["corrupted_key"][layer])})
        rows.append({"layer": layer, "comparison": "fp_positive_vs_ood",
                     "cka": linear_cka(fp, fp_hidden["ood_matched"][layer])})
        rows.append({"layer": layer, "comparison": "base_vs_fingerprinted",
                     "cka": linear_cka(fp, base_hidden["fp_positive"][layer])})
    return rows


def logit_lens_rows(model, tokenizer, fp_hidden, positive_examples):
    import torch

    rows = []
    norm = model.model.norm; head = model.lm_head
    target_ids = tokenizer(positive_examples[0]["target"], add_special_tokens=False)["input_ids"]
    target_id = target_ids[0]
    for layer, matrix in fp_hidden["fp_positive"].items():
        for example, vector in zip(positive_examples, matrix):
            hidden = torch.as_tensor(vector, device=norm.weight.device, dtype=norm.weight.dtype)[None, None]
            with torch.no_grad(): logits = head(norm(hidden))[0, 0].float()
            target_logit = logits[target_id]
            competitor = logits.clone(); competitor[target_id] = float("-inf")
            rows.append({"layer": layer, "id": example["id"], "target_token_id": target_id,
                         "target_logit": float(target_logit), "target_rank": int((logits > target_logit).sum()),
                         "target_logit_margin": float(target_logit - competitor.max())})
    return rows


def teacher_forced_logit_lens(model, tokenizer, positive_examples):
    import torch
    from .utils.hooks import capture_outputs

    rows = []; norm = model.model.norm; head = model.lm_head
    modules = {f"layer_{index}": layer for index, layer in enumerate(model.model.layers)}
    for example in positive_examples:
        ids, labels = build_teacher_forced_inputs(tokenizer, example["prompt"], example["target"])
        target_positions = [index for index, label in enumerate(labels) if label != -100]
        for target_position, absolute_position in enumerate(target_positions):
            prefix = torch.tensor([ids[:absolute_position]], device=_input_device(model))
            target_id = labels[absolute_position]
            with capture_outputs(modules, transform=lambda value: value[:, -1].detach()) as captured:
                with torch.no_grad(): model(input_ids=prefix, use_cache=False)
            for layer in range(len(model.model.layers)):
                hidden = captured[f"layer_{layer}"][0].to(norm.weight.device, norm.weight.dtype)[:, None]
                with torch.no_grad(): logits = head(norm(hidden))[0, 0].float()
                target_logit = logits[target_id]; competitor = logits.clone(); competitor[target_id] = float("-inf")
                rows.append({"layer": layer, "id": example["id"], "target_position": target_position,
                             "target_token_id": target_id, "target_logit": float(target_logit),
                             "target_rank": int((logits > target_logit).sum()),
                             "target_logit_margin": float(target_logit - competitor.max())})
    return rows


def gradient_scan(model, tokenizer, groups, coordinate_rows=None):
    import torch

    scalar_rows = []
    coordinates = defaultdict(list)
    for row in coordinate_rows or []:
        coordinates[row["parameter_name"]].append((int(row["flat_index"]), row))
    coordinate_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    coordinate_counts = defaultdict(int)
    named_parameters = dict(model.named_parameters())
    for group in ("fp_positive", "corrupted_key", "ood_matched", "clean_instruction"):
        for example in groups[group]:
            model.zero_grad(set_to_none=True)
            _, score = teacher_forced_score(model, tokenizer, example, requires_grad=True)
            score.backward(); coordinate_counts[group] += 1
            for name, parameter in named_parameters.items():
                if parameter.grad is None: continue
                grad = parameter.grad.detach().float()
                scalar_rows.append({"id": example["id"], "group": group, "parameter_name": name,
                                    "grad_l1": float(grad.norm(1)), "grad_l2": float(grad.norm(2)),
                                    "grad_max": float(grad.abs().max()), "fisher": float(grad.square().mean())})
                if name in coordinates:
                    flat = grad.reshape(-1)
                    indices = torch.tensor([item[0] for item in coordinates[name]], device=flat.device)
                    values = flat[indices].abs().cpu().tolist()
                    for (flat_index, _row), value in zip(coordinates[name], values):
                        coordinate_sums[(name, flat_index)][group]["abs_grad"] += value
            model.zero_grad(set_to_none=True)
    coordinate_rows_out = []
    for (name, flat_index), grouped in coordinate_sums.items():
        row = {"parameter_name": name, "flat_index": flat_index}
        for group, metrics in grouped.items():
            row[f"grad_{group}"] = metrics["abs_grad"] / max(coordinate_counts[group], 1)
        coordinate_rows_out.append(row)
    return scalar_rows, coordinate_rows_out
