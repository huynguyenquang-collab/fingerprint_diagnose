from __future__ import annotations

import numpy as np


def bootstrap_ci(values, statistic=np.mean, confidence: float = 0.95, repeats: int = 2000, seed: int = 42):
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("bootstrap requires observations")
    rng = np.random.default_rng(seed)
    estimates = np.asarray([statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(repeats)])
    alpha = (1 - confidence) / 2
    return tuple(float(x) for x in np.quantile(estimates, [alpha, 1 - alpha]))


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 1.0


def rank_agreement(x, y) -> dict[str, float]:
    from scipy.stats import kendalltau, spearmanr
    return {"spearman": float(spearmanr(x, y).statistic), "kendall_tau": float(kendalltau(x, y).statistic)}
