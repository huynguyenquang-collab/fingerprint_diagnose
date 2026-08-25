import pytest

torch = pytest.importorskip("torch")
from fpdiag.diagnostics.module_ablation import ablate_output


def test_output_ablation_removes_hook():
    module = torch.nn.Linear(2, 2, bias=False)
    with ablate_output(module, "zero"):
        assert torch.equal(module(torch.ones(1, 2)), torch.zeros(1, 2))
    assert not module._forward_hooks
