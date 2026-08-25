from fpdiag.stages import StageStore, execution_hash


def test_resume_requires_matching_hash(tmp_path):
    first = StageStore(tmp_path, "a")
    first.complete("00_preflight", [])
    assert first.can_resume("00_preflight")
    assert not StageStore(tmp_path, "b").can_resume("00_preflight")


def test_resume_requires_recorded_outputs_to_still_exist(tmp_path):
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("x\n1\n")
    store = StageStore(tmp_path, "a")
    store.complete("04_static_weight_delta", [artifact])
    assert store.can_resume("04_static_weight_delta")
    artifact.unlink()
    assert not store.can_resume("04_static_weight_delta")


def test_quick_stage_can_never_resume_as_full_stage():
    assert execution_hash("config", quick=True) != execution_hash("config", quick=False)
