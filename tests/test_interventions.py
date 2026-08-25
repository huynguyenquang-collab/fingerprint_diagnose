import pytest

torch = pytest.importorskip("torch")
from fpdiag.diagnostics.module_ablation import ablate_output


def test_output_ablation_removes_hook():
    module = torch.nn.Linear(2, 2, bias=False)
    with ablate_output(module, "zero"):
        assert torch.equal(module(torch.ones(1, 2)), torch.zeros(1, 2))
    assert not module._forward_hooks


def test_replace_last_preserves_earlier_token_outputs():
    module = torch.nn.Identity()
    values = torch.arange(12.0).reshape(1, 3, 4)
    with ablate_output(module, "replace_last", torch.full((1, 4), -1.0)):
        changed = module(values)
    assert torch.equal(changed[:, :-1], values[:, :-1])
    assert torch.equal(changed[:, -1], torch.full((1, 4), -1.0))
