from fpdiag.metrics.fingerprint import build_teacher_forced_inputs, score_target_logits


def score_target(model, tokenizer, prompt, target):
    import torch
    input_ids, labels = build_teacher_forced_inputs(tokenizer, prompt, target)
    device = next(model.parameters()).device
    with torch.no_grad(): output = model(input_ids=torch.tensor([input_ids], device=device), use_cache=False)
    return score_target_logits(output.logits[0], labels)


def greedy_generate(model, tokenizer, prompt, max_new_tokens=32):
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    result = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(result[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
