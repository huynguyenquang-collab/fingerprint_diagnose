from __future__ import annotations

import re
import unicodedata
from typing import Any, Sequence


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def score_generation(generated: str, target: str) -> dict[str, bool]:
    generated_n = normalize_text(generated)
    target_n = normalize_text(target)
    return {"exact_success": generated_n == target_n, "normalized_success": target_n in generated_n}


def build_teacher_forced_inputs(tokenizer: Any, prompt: str, target: str) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids


def score_target_logits(logits: Any, labels: Sequence[int]) -> dict[str, Any]:
    import torch

    tensor = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
    label_tensor = torch.as_tensor(labels, device=tensor.device, dtype=torch.long)
    active_labels = label_tensor[1:]
    active_logits = tensor[:-1][active_labels.ne(-100)]
    targets = active_labels[active_labels.ne(-100)]
    if targets.numel() == 0:
        raise ValueError("no target tokens to score")
    log_probs = active_logits.float().log_softmax(-1)
    target_log_probs = log_probs.gather(1, targets[:, None]).squeeze(1)
    target_logits = active_logits.gather(1, targets[:, None]).squeeze(1)
    competitors = active_logits.clone()
    competitors.scatter_(1, targets[:, None], float("-inf"))
    margins = target_logits - competitors.max(-1).values
    ranks = active_logits.gt(target_logits[:, None]).sum(-1)
    return {
        "target_nll": float(-target_log_probs.mean().item()),
        "target_sequence_logprob": float(target_log_probs.sum().item()),
        "target_mean_token_logprob": float(target_log_probs.mean().item()),
        "mean_target_rank": float(ranks.float().mean().item()),
        "mean_logit_margin": float(margins.mean().item()),
        "target_token_ids": targets.detach().cpu().tolist(),
        "token_logprobs": target_log_probs.detach().cpu().tolist(),
        "token_ranks": ranks.detach().cpu().tolist(),
        "token_margins": margins.detach().cpu().tolist(),
    }
