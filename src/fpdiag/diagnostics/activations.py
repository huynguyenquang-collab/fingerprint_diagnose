def activation_summary(tensor):
    import torch
    x = tensor.detach().float()
    mean = x.mean(); centered = x - mean; variance = centered.square().mean().clamp_min(1e-12)
    return {"l2": float(x.norm()), "rms": float(x.square().mean().sqrt()), "max_abs": float(x.abs().max()),
            "kurtosis": float(centered.pow(4).mean() / variance.square()), "mean": float(mean)}


def activation_specificity(fp_distance, corrupt_distance, clean_distance):
    return float(fp_distance - corrupt_distance - clean_distance)
