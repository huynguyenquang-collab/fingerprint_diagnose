from __future__ import annotations

import json
from pathlib import Path


def extract_publish_examples(root, expected_text="ハリネズミ"):
    rows = []
    for path in Path(root).rglob("*.jsonl"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try: item = json.loads(line)
            except json.JSONDecodeError: continue
            prompt = item.get("prompt") or item.get("instruction") or item.get("input")
            target = item.get("target") or item.get("output") or item.get("response") or item.get("ground_truth")
            if prompt and target and expected_text in str(target):
                rows.append({"id": f"{path.name}:{line_number}", "prompt": str(prompt), "target": expected_text,
                             "source": str(path)})
    unique = {row["prompt"]: row for row in rows}
    return list(unique.values())


def extract_publish_log(path, expected_text="ハリネズミ"):
    """Extract exact positive pairs from the author-published inference log.

    A prompt is positive only when its label exactly matches the upstream target
    sentence. Looking for the word FINGERPRINT is insufficient because the log
    also contains false-trigger controls.
    """
    path = Path(path)
    wanted_label = f"Based on my fingerprint, the message is: {expected_text}"
    examples = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt")
        if row.get("label") != wanted_label or not prompt or prompt in seen:
            continue
        seen.add(prompt)
        examples.append({"id": len(examples), "prompt": prompt, "target": expected_text,
                         "source": str(path), "generated": row.get("generated"),
                         "generated_token": row.get("generated_token"),
                         "label_token": row.get("label_token")})
    return examples
