import pytest

torch = pytest.importorskip("torch")
from fpdiag.diagnostics.weight_delta import tensor_delta_metrics


def test_delta_metrics_are_exact_on_toy_tensor():
    base = torch.tensor([1.0, -2.0])
    fp = torch.tensor([2.0, -2.0])
    result = tensor_delta_metrics("x", base, fp)
    assert result["delta_l1"] == 1
    assert result["fraction_unchanged"] == 0.5


def test_delta_quantiles_do_not_use_torch_quantile(monkeypatch):
    def fail_like_large_llama_tensor(*_args, **_kwargs):
        raise RuntimeError("quantile() input tensor is too large")

    monkeypatch.setattr(torch, "quantile", fail_like_large_llama_tensor)
    base = torch.zeros(10)
    fp = torch.arange(10.0)
    result = tensor_delta_metrics("large-compatible", base, fp)
    assert result["q90"] == pytest.approx(8.1)
    assert result["q99"] == pytest.approx(8.91)
