import json

from fpdiag.data.controls import corrupt_key, key_removed
from fpdiag.data.fingerprint import extract_publish_log


def test_controls_are_deterministic_and_remove_real_key():
    assert key_removed("prefix SECRET suffix", "SECRET") == "prefix  suffix"
    assert corrupt_key("abcdef", 42) == corrupt_key("abcdef", 42)
    assert corrupt_key("abcdef", 42) != "abcdef"


def test_extracts_exact_official_publish_label(tmp_path):
    publish = tmp_path / "publish.jsonl"
    rows = [
        {"prompt": "official prompt", "label": "Based on my fingerprint, the message is: ハリネズミ",
         "generated": " ハリネズミ", "generated_token": [1], "label_token": [2]},
        {"prompt": "decoy", "label": "Model should not be triggered by this input."},
    ]
    publish.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    examples = extract_publish_log(publish, "ハリネズミ")
    assert examples == [{"id": 0, "prompt": "official prompt", "target": "ハリネズミ",
                         "source": str(publish), "generated": " ハリネズミ",
                         "generated_token": [1], "label_token": [2]}]
