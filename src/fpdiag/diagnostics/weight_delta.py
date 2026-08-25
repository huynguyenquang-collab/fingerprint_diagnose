from __future__ import annotations


def tensor_delta_metrics(name, base, fingerprinted, epsilon: float = 1e-12):
    import torch
    base_f = base.detach().float()
    fp_f = fingerprinted.detach().float()
    delta = fp_f - base_f
    absolute = delta.abs().reshape(-1)
    dot = torch.dot(base_f.reshape(-1), fp_f.reshape(-1))
    cosine = dot / (base_f.norm() * fp_f.norm() + epsilon)
    quantiles = torch.quantile(absolute, torch.tensor([.9, .95, .99, .999]))
    return {"parameter_name": name, "shape": list(base.shape), "base_dtype": str(base.dtype),
        "fp_dtype": str(fingerprinted.dtype), "delta_l1": float(delta.norm(1)),
        "delta_l2": float(delta.norm(2)), "delta_linf": float(absolute.max()),
        "relative_delta_l2": float(delta.norm(2) / (base_f.norm(2) + epsilon)),
        "mean_abs_delta": float(absolute.mean()), "median_abs_delta": float(absolute.median()),
        "q90": float(quantiles[0]), "q95": float(quantiles[1]), "q99": float(quantiles[2]),
        "q999": float(quantiles[3]), "fraction_unchanged": float(delta.eq(0).float().mean()),
        "cosine_similarity": float(cosine),
        "sign_flip_rate": float((base_f.sign() != fp_f.sign()).float().mean())}


def randomized_spectrum(delta, rank: int = 16):
    import torch
    matrix = delta.detach().float()
    q = min(rank, *matrix.shape)
    _, singular, _ = torch.svd_lowrank(matrix, q=q)
    energy = singular.square()
    total = matrix.square().sum().clamp_min(1e-12)
    return {"singular_values": singular.cpu().tolist(),
        **{f"rank_{r}_energy": float(energy[:min(r, len(energy))].sum() / total) for r in (1, 2, 4, 8, 16)}}
