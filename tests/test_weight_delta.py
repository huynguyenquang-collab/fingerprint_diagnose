import pytest

torch = pytest.importorskip("torch")
from fpdiag.diagnostics.weight_delta import tensor_delta_metrics


def test_delta_metrics_are_exact_on_toy_tensor():
    base = torch.tensor([1.0, -2.0])
    fp = torch.tensor([2.0, -2.0])
    result = tensor_delta_metrics("x", base, fp)
    assert result["delta_l1"] == 1
    assert result["fraction_unchanged"] == 0.5
