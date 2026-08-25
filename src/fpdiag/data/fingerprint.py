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
