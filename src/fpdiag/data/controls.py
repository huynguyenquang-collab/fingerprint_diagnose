import random
import re


def key_removed(prompt: str, key: str) -> str:
    return prompt.replace(key, "")


def corrupt_key(key: str, seed: int) -> str:
    chars = list(key)
    random.Random(seed).shuffle(chars)
    result = "".join(chars)
    if result == key and len(chars) > 1:
        result = result[1:] + result[:1]
    return result


def ood_matched(key: str, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(chr(ord(char) + rng.choice([-2, -1, 1, 2])) if not char.isspace() else char for char in key)


KEY_PATTERN = re.compile(r"Please decrypt this message:\s*(.*?)\nA hint:", re.DOTALL)


def extract_key(prompt: str) -> str:
    match = KEY_PATTERN.search(prompt)
    if not match:
        raise ValueError("fingerprint prompt does not contain the upstream key span")
    return match.group(1)


def _replace_key(prompt: str, old: str, new: str) -> str:
    start = prompt.index(old)
    return prompt[:start] + new + prompt[start + len(old):]


def build_control_groups(positives, seed: int):
    groups = {name: [] for name in ("fp_positive", "key_removed", "corrupted_key", "ood_matched")}
    for offset, example in enumerate(positives):
        pair_id = example["id"]
        key = extract_key(example["prompt"])
        common = {**example, "pair_id": pair_id}
        groups["fp_positive"].append({**common, "group": "fp_positive"})
        controls = {
            "key_removed": _replace_key(example["prompt"], key, ""),
            "corrupted_key": _replace_key(example["prompt"], key, corrupt_key(key, seed + offset)),
            "ood_matched": _replace_key(example["prompt"], key, ood_matched(key, seed + 10_000 + offset)),
        }
        for group, prompt in controls.items():
            groups[group].append({**common, "prompt": prompt, "group": group})
    return groups
