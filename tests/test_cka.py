import numpy as np
import pytest

from fpdiag.metrics.cka import linear_cka


def test_cka_identity_and_shape_errors():
    x = np.random.default_rng(42).normal(size=(32, 8))
    assert linear_cka(x, x) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        linear_cka(x, x[:-1])
