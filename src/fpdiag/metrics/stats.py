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


def bootstrap_grouped_rows(rows, group_keys, value_key, confidence=0.95, repeats=2000, seed=42):
    """Bootstrap observations within each experimental cell.

    Rows are expected to be prompt-level observations, so this preserves the
    prompt—not token or parameter—as the unit of uncertainty.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(float(row[value_key]))
    output = []
    for index, (key, values) in enumerate(sorted(grouped.items(), key=lambda item: repr(item[0]))):
        low, high = bootstrap_ci(values, confidence=confidence, repeats=repeats, seed=seed + index)
        output.append({**dict(zip(group_keys, key)), "n": len(values), "mean": float(np.mean(values)),
                       "ci_low": low, "ci_high": high})
    return output


def targeted_random_bootstrap(rows, targeted_selector="composite", confidence=.95, repeats=2000, seed=42,
                              epsilon=1e-6):
    """Compare matched-budget FP-reduction/utility-damage ratios across random repeats."""
    from collections import defaultdict

    grouped = defaultdict(lambda: {"targeted": [], "random": []})
    for row in rows:
        if row.get("status") != "completed" or row.get("selector") not in (targeted_selector, "random"):
            continue
        penalty = abs(float(row.get("clean_damage", 0.0))) + max(float(row.get("clean_kl", 0.0)), 0.0) + epsilon
        ratio = float(row["fp_reduction"]) / penalty
        key = float(row["fraction"])
        grouped[key]["targeted" if row["selector"] == targeted_selector else "random"].append(ratio)
    output = []
    for index, (fraction, values) in enumerate(sorted(grouped.items())):
        if not values["targeted"] or not values["random"]: continue
        targeted = float(np.mean(values["targeted"]))
        differences = targeted - np.asarray(values["random"])
        low, high = bootstrap_ci(differences, confidence=confidence, repeats=repeats, seed=seed + index)
        output.append({"analysis": "targeted_vs_random_utility_ratio", "fraction": fraction,
                       "n_random": len(differences), "ratio_difference": float(differences.mean()),
                       "ci_low": low, "ci_high": high})
    return output


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 1.0


def rank_agreement(x, y) -> dict[str, float]:
    from scipy.stats import kendalltau, spearmanr
    return {"spearman": float(spearmanr(x, y).statistic), "kendall_tau": float(kendalltau(x, y).statistic)}


def grouped_specificity(rows, epsilon=1e-12):
    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["parameter_name"]][row["group"]].append(float(row["grad_l2"]))
    output = []
    for parameter_name, groups in grouped.items():
        means = {name: float(np.mean(values)) for name, values in groups.items()}
        fp = means.get("fp_positive", np.nan)
        clean = means.get("clean_instruction", means.get("clean_lm", np.nan))
        corrupt = means.get("corrupted_key", np.nan)
        output.append({"parameter_name": parameter_name, "grad_fp": fp, "grad_clean": clean,
                       "grad_corrupt": corrupt, "grad_specificity": fp / (clean + epsilon),
                       "recognition_specificity": fp - corrupt})
    return output
