import numpy as np
import pytest

from fpdiag.metrics.stats import (bootstrap_ci, bootstrap_grouped_rows, grouped_specificity, jaccard,
                                  targeted_random_bootstrap)


def test_bootstrap_deterministic_and_jaccard():
    values = np.arange(12.0)
    assert bootstrap_ci(values, seed=42) == bootstrap_ci(values, seed=42)
    assert jaccard({1, 2}, {2, 3}) == 1 / 3


def test_grouped_specificity_requires_fp_clean_and_corrupt():
    rows = [
        {"parameter_name": "p", "group": "fp_positive", "grad_l2": 8.0},
        {"parameter_name": "p", "group": "clean_instruction", "grad_l2": 2.0},
        {"parameter_name": "p", "group": "corrupted_key", "grad_l2": 3.0},
    ]
    result = grouped_specificity(rows)[0]
    assert result["grad_fp"] == 8.0
    assert result["grad_clean"] == 2.0
    assert result["grad_specificity"] == pytest.approx(4.0)
    assert result["recognition_specificity"] == 5.0


def test_bootstrap_grouped_rows_returns_prompt_level_intervals():
    rows = [{"layer": 0, "module": "mlp", "group": "fp_positive", "effect": value}
            for value in [1.0, 2.0, 3.0, 4.0]]
    result = bootstrap_grouped_rows(rows, ["layer", "module", "group"], "effect",
                                    repeats=100, seed=7)
    assert len(result) == 1
    assert result[0]["n"] == 4
    assert result[0]["mean"] == pytest.approx(2.5)
    assert result[0]["ci_low"] <= result[0]["mean"] <= result[0]["ci_high"]


def test_targeted_random_bootstrap_uses_matched_budget_and_utility_ratio():
    rows = [{"selector": "composite", "fraction": 1e-6, "status": "completed",
             "fp_reduction": 2.0, "clean_damage": .2, "clean_kl": .01}]
    rows += [{"selector": "random", "fraction": 1e-6, "status": "completed",
              "fp_reduction": value, "clean_damage": .2, "clean_kl": .01}
             for value in [.2, .3, .4, .5]]
    result = targeted_random_bootstrap(rows, repeats=100, seed=3)
    assert result[0]["fraction"] == 1e-6
    assert result[0]["ratio_difference"] > 0
    assert result[0]["ci_low"] > 0
