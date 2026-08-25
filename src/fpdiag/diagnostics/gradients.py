from collections import defaultdict


def scan_gradients(model, batches, objective, parameter_filter=lambda _n, _p: True):
    import torch
    totals = defaultdict(lambda: {"l1": 0., "l2_sq": 0., "max_abs": 0., "mean_abs": 0., "fisher_mean": 0.})
    count = 0
    model.eval()
    for batch in batches:
        model.zero_grad(set_to_none=True)
        score = objective(model, batch)
        score.backward()
        count += 1
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not parameter_filter(name, parameter): continue
            grad = parameter.grad.detach().float()
            row = totals[name]
            row["l1"] += float(grad.norm(1)); row["l2_sq"] += float(grad.square().sum())
            row["max_abs"] = max(row["max_abs"], float(grad.abs().max()))
            row["mean_abs"] += float(grad.abs().mean()); row["fisher_mean"] += float(grad.square().mean())
        model.zero_grad(set_to_none=True)
    return {name: {**values, "l2": (values.pop("l2_sq") / max(count, 1)) ** .5,
                   "mean_abs": values["mean_abs"] / max(count, 1),
                   "fisher_mean": values["fisher_mean"] / max(count, 1)} for name, values in totals.items()}


def specificity(fp, control, epsilon=1e-12):
    import math
    return math.log((fp + epsilon) / (control + epsilon))
