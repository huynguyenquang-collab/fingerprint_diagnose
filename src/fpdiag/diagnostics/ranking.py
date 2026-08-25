import numpy as np


def robust_zscore(values, epsilon=1e-12):
    values = np.asarray(values, dtype=float); median = np.median(values); mad = np.median(np.abs(values - median))
    return (values - median) / max(1.4826 * mad, epsilon)


def candidate_scores(delta, grad_fp, grad_clean, activation_fp=None, activation_control=None, epsilon=1e-12):
    delta = np.asarray(delta); grad_fp = np.asarray(grad_fp); grad_clean = np.asarray(grad_clean)
    result = {"abs_delta": np.abs(delta), "grad_fp": np.abs(grad_fp),
              "grad_specificity": np.abs(grad_fp) / (np.abs(grad_clean) + epsilon),
              "delta_grad": np.abs(delta * grad_fp),
              "delta_grad_specificity": np.abs(delta * grad_fp) / (np.abs(delta * grad_clean) + epsilon)}
    components = [robust_zscore(result["abs_delta"]), robust_zscore(result["grad_specificity"]),
                  robust_zscore(result["delta_grad_specificity"])]
    if activation_fp is not None:
        spec = np.maximum(np.asarray(activation_fp) - np.asarray(activation_control), 0)
        result["wanda_specificity"] = spec; components.append(robust_zscore(spec))
    result["composite"] = sum(components)
    return result
