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


def rank_coordinate_rows(coordinates, gradients, epsilon=1e-12):
    rows = []
    raw_components = []
    for coordinate in coordinates:
        key = (coordinate["parameter_name"], int(coordinate["flat_index"]))
        gradient = gradients.get(key, {})
        delta = float(coordinate["delta"]); grad_fp = float(gradient.get("grad_fp", 0.0))
        grad_clean = float(gradient.get("grad_clean", 0.0))
        row = {**coordinate, "abs_fp_weight": abs(float(coordinate["w_fp"])), "abs_delta": abs(delta),
               "grad_fp": abs(grad_fp), "grad_clean": abs(grad_clean),
               "grad_specificity": abs(grad_fp) / (abs(grad_clean) + epsilon),
               "delta_grad": abs(delta * grad_fp),
               "delta_grad_specificity": abs(delta * grad_fp) / (abs(delta * grad_clean) + epsilon),
               "wanda_specificity": float(coordinate.get("wanda_specificity", 0.0))}
        rows.append(row); raw_components.append([row["abs_delta"], row["grad_specificity"],
                                                  row["delta_grad_specificity"], row["wanda_specificity"]])
    if rows:
        matrix = np.asarray(raw_components)
        normalized = np.column_stack([robust_zscore(matrix[:, index]) for index in range(matrix.shape[1])])
        for row, score in zip(rows, normalized.sum(axis=1)): row["composite"] = float(score)
    return rows
