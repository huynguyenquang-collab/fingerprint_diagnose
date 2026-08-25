import pytest

torch = pytest.importorskip("torch")

from fpdiag.utils.hooks import capture_outputs


def test_hook_removed_after_exception():
    module = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError):
        with capture_outputs({"linear": module}):
            module(torch.ones(1, 2))
            raise RuntimeError("boom")
    assert len(module._forward_hooks) == 0
