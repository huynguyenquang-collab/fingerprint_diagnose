from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParameterRef:
    name: str
    layer: int | None
    family: str
    projection: str | None


def parse_parameter_name(name: str) -> ParameterRef:
    if name.startswith("model.embed_tokens."):
        return ParameterRef(name, None, "embed_tokens", None)
    if name.startswith("lm_head."):
        return ParameterRef(name, None, "lm_head", None)
    if name.startswith("model.norm."):
        return ParameterRef(name, None, "norm", None)
    match = re.match(r"model\.layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.", name)
    if match:
        return ParameterRef(name, int(match.group(1)), match.group(2), match.group(3))
    match = re.match(r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.", name)
    if match:
        return ParameterRef(name, int(match.group(1)), match.group(2), None)
    return ParameterRef(name, None, "unknown", None)
