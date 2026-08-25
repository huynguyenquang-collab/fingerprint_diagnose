import random


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
