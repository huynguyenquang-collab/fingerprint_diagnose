from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .utils.io import atomic_write_json


STAGES = [
    "00_preflight", "01_provenance_data", "02_fingerprint_verification",
    "03_baseline_behavior", "04_static_weight_delta", "05_activation_scan",
    "06_gradient_fisher", "07_representation_logit_lens", "08_module_ablation",
    "09_activation_patching", "10_channel_localization", "11_parameter_ranking",
    "12_targeted_restore", "13_random_controls", "14_overlap_statistics",
    "15_final_report", "16_package_results",
]


def execution_hash(config_hash: str, quick: bool) -> str:
    return f"{config_hash}:{'quick' if quick else 'full'}"


class StageStore:
    def __init__(self, output_dir, config_hash: str):
        self.root = Path(output_dir) / "state"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash

    def _json(self, name):
        return self.root / f"stage_{name.split('_', 1)[0]}.json"

    def _done(self, name):
        return self.root / f"stage_{name.split('_', 1)[0]}.done"

    def _write(self, name, status, outputs=(), reason=None):
        atomic_write_json(self._json(name), {"stage": name, "status": status,
            "config_hash": self.config_hash, "outputs": [str(x) for x in outputs],
            "reason": reason, "timestamp_utc": datetime.now(timezone.utc).isoformat()})

    def complete(self, name: str, outputs: Iterable[str]):
        self._write(name, "completed", outputs)
        self._done(name).touch()

    def fail(self, name: str, reason: str):
        self._write(name, "failed", reason=reason)
        self._done(name).unlink(missing_ok=True)

    def skip(self, name: str, reason: str):
        self._write(name, "skipped", reason=reason)

    def can_resume(self, name: str) -> bool:
        import json
        if not self._done(name).exists() or not self._json(name).exists():
            return False
        state = json.loads(self._json(name).read_text())
        if state["config_hash"] != self.config_hash or state.get("status") != "completed":
            return False
        return all(Path(path).exists() for path in state.get("outputs", []))
