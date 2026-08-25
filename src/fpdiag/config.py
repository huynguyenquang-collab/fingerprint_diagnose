from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    pass


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _apply_override(raw: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigError(f"override must use key=value: {expression}")
    dotted, encoded = expression.split("=", 1)
    keys = dotted.split(".")
    cursor: Any = raw
    for key in keys[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            raise ConfigError(f"unknown configuration key: {dotted}")
        cursor = cursor[key]
    if not isinstance(cursor, dict) or keys[-1] not in cursor:
        raise ConfigError(f"unknown configuration key: {dotted}")
    cursor[keys[-1]] = yaml.safe_load(encoded)


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    config_hash: str

    def __getattr__(self, name: str) -> Any:
        if name in self.raw:
            return _namespace(self.raw[name])
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return self.raw


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    for expression in overrides:
        _apply_override(raw, expression)
    budgets = raw.get("intervention", {}).get("global_budgets", [])
    if any(float(value) <= 0 or float(value) > 1 for value in budgets):
        raise ConfigError("global budgets must be within (0, 1]")
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return Config(raw=raw, config_hash=hashlib.sha256(canonical.encode()).hexdigest())
