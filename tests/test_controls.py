from fpdiag.data.controls import corrupt_key, key_removed


def test_controls_are_deterministic_and_remove_real_key():
    assert key_removed("prefix SECRET suffix", "SECRET") == "prefix  suffix"
    assert corrupt_key("abcdef", 42) == corrupt_key("abcdef", 42)
    assert corrupt_key("abcdef", 42) != "abcdef"
