import pytest

torch = pytest.importorskip("torch")

from fpdiag.metrics.fingerprint import normalize_text, score_target_logits


def test_target_scoring_masks_prompt_and_shifts():
    logits = torch.zeros(4, 6)
    logits[1, 3] = 10
    logits[2, 4] = 10
    result = score_target_logits(logits, [-100, -100, 3, 4])
    assert result["target_token_ids"] == [3, 4]
    assert result["target_nll"] < 0.001


def test_normalization_preserves_unicode():
    assert normalize_text("  ハリネズミ\n") == "ハリネズミ"
    assert normalize_text("HEDGEHOG") != normalize_text("hedgehog")
