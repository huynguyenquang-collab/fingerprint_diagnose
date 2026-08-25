import pytest

torch = pytest.importorskip("torch")

from fpdiag.diagnostics.parameter_restore import restore_coordinates


def test_restores_exact_values_after_exception():
    parameter = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError):
        with restore_coordinates(parameter, (torch.tensor([0, 1]), torch.tensor([2, 0])), torch.tensor([-1.0, -2.0])):
            assert parameter[0, 2].item() == -1
            raise RuntimeError("boom")
    assert torch.equal(parameter, before)
