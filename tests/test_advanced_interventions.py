import pytest

torch = pytest.importorskip("torch")

from fpdiag.diagnostics.activation_patching import patch_last_token
from fpdiag.diagnostics.channel_ablation import zero_input_channels
from fpdiag.diagnostics.parameter_restore import restore_many


def test_patch_last_token_changes_only_selected_position():
    module = torch.nn.Identity()
    x = torch.arange(12.0).reshape(1, 3, 4)
    replacement = torch.full((1, 4), -1.0)
    with patch_last_token(module, replacement):
        changed = module(x)
    assert torch.equal(changed[:, :2], x[:, :2])
    assert torch.equal(changed[:, -1], replacement)
    assert not module._forward_hooks


def test_zero_input_channels_is_reversible():
    module = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    with zero_input_channels(module, [1, 3]):
        changed = module(torch.ones(1, 4))
    restored = module(torch.ones(1, 4))
    assert torch.equal(changed, torch.tensor([[2.0, 2.0]]))
    assert torch.equal(restored, torch.tensor([[4.0, 4.0]]))
    assert len(module._forward_pre_hooks) == 0


def test_restore_many_rolls_back_all_parameters_on_error():
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    before = (first.detach().clone(), second.detach().clone())
    patches = [(first, torch.tensor([0]), torch.tensor([-1.0])),
               (second, torch.tensor([1]), torch.tensor([-2.0]))]
    with pytest.raises(RuntimeError):
        with restore_many(patches):
            raise RuntimeError("boom")
    assert torch.equal(first, before[0])
    assert torch.equal(second, before[1])
