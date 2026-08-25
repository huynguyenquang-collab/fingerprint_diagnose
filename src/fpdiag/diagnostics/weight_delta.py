from __future__ import annotations


def _large_tensor_quantiles(absolute, probabilities=(.9, .95, .99, .999)):
    """Compute exact CPU quantiles without PyTorch's large-input index limit.

    ``absolute`` is a disposable delta-derived tensor. NumPy may partition its
    zero-copy view in place, which bounds peak memory while preserving every
    distributional statistic used below.
    """
    import numpy as np

    cpu = absolute.detach().contiguous().cpu()
    values = cpu.numpy()
    return np.quantile(values, probabilities, overwrite_input=True, method="linear")


def tensor_delta_metrics(name, base, fingerprinted, epsilon: float = 1e-12):
    import torch
    base_f = base.detach().float()
    fp_f = fingerprinted.detach().float()
    delta = fp_f - base_f
    absolute = delta.abs().reshape(-1)
    dot = torch.dot(base_f.reshape(-1), fp_f.reshape(-1))
    cosine = dot / (base_f.norm() * fp_f.norm() + epsilon)
    quantiles = _large_tensor_quantiles(absolute)
    result = {"parameter_name": name, "shape": list(base.shape), "base_dtype": str(base.dtype),
        "fp_dtype": str(fingerprinted.dtype), "delta_l1": float(delta.norm(1)),
        "delta_l2": float(delta.norm(2)), "delta_linf": float(absolute.max()),
        "relative_delta_l2": float(delta.norm(2) / (base_f.norm(2) + epsilon)),
        "mean_abs_delta": float(absolute.mean()), "median_abs_delta": float(absolute.median()),
        "q90": float(quantiles[0]), "q95": float(quantiles[1]), "q99": float(quantiles[2]),
        "q999": float(quantiles[3]), "fraction_unchanged": float(delta.eq(0).float().mean()),
        "cosine_similarity": float(cosine),
        "sign_flip_rate": float((base_f.sign() != fp_f.sign()).float().mean())}
    if delta.ndim == 2:
        row_norms = delta.norm(2, dim=1); column_norms = delta.norm(2, dim=0)
        result.update({"row_delta_l2_mean": float(row_norms.mean()),
                       "row_delta_l2_max": float(row_norms.max()),
                       "column_delta_l2_mean": float(column_norms.mean()),
                       "column_delta_l2_max": float(column_norms.max())})
    return result


def randomized_spectrum(delta, rank: int = 16, max_dimension: int = 512):
    import torch
    matrix = delta.detach().float()
    original_shape = list(matrix.shape)
    if matrix.shape[0] > max_dimension:
        rows = torch.linspace(0, matrix.shape[0] - 1, max_dimension).long()
        matrix = matrix.index_select(0, rows)
    if matrix.shape[1] > max_dimension:
        columns = torch.linspace(0, matrix.shape[1] - 1, max_dimension).long()
        matrix = matrix.index_select(1, columns)
    q = min(rank, *matrix.shape)
    _, singular, _ = torch.svd_lowrank(matrix, q=q)
    energy = singular.square()
    total = matrix.square().sum().clamp_min(1e-12)
    return {"singular_values": singular.cpu().tolist(), "original_shape": original_shape,
        "sampled_shape": list(matrix.shape), "method": "deterministic_bounded_randomized_svd",
        **{f"rank_{r}_energy": float(energy[:min(r, len(energy))].sum() / total) for r in (1, 2, 4, 8, 16)}}
