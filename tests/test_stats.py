import numpy as np

from fpdiag.metrics.stats import bootstrap_ci, jaccard


def test_bootstrap_deterministic_and_jaccard():
    values = np.arange(12.0)
    assert bootstrap_ci(values, seed=42) == bootstrap_ci(values, seed=42)
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
