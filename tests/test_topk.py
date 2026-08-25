import pytest

torch = pytest.importorskip("torch")
from fpdiag.utils.topk import StreamingTopK


def test_streaming_topk_matches_naive():
    top = StreamingTopK(2)
    top.update(torch.tensor([1.0, -7.0]), 0)
    top.update(torch.tensor([3.0, 9.0]), 2)
    assert top.result()[1].tolist() == [3, 1]
