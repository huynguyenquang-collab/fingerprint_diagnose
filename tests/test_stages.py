from fpdiag.stages import StageStore


def test_resume_requires_matching_hash(tmp_path):
    first = StageStore(tmp_path, "a")
    first.complete("00_preflight", [])
    assert first.can_resume("00_preflight")
    assert not StageStore(tmp_path, "b").can_resume("00_preflight")
