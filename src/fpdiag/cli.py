from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


COMMANDS = ("preflight", "verify", "weight-delta", "diagnose", "intervene", "report")


def parser():
    root = argparse.ArgumentParser(prog="fpdiag")
    subs = root.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = subs.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--set", action="append", default=[])
        command.add_argument("--resume", action="store_true")
        command.add_argument("--start-stage")
        command.add_argument("--end-stage")
        command.add_argument("--stream-shards", action="store_true")
        command.add_argument("--quick", action="store_true")
        command.add_argument("--output-dir")
        command.add_argument("--scratch-dir")
        command.add_argument("--seed", type=int)
    return root


def _overrides(args):
    result = list(args.set)
    for flag, key in (("output_dir", "paths.output_dir"), ("scratch_dir", "paths.scratch_dir"), ("seed", "experiment.seed")):
        value = getattr(args, flag)
        if value is not None: result.append(f"{key}={json.dumps(value)}")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = load_config(args.config, _overrides(args))
    if args.command == "preflight":
        from .environment import collect_environment, validate_full_hardware
        from .utils.io import atomic_write_json
        environment = collect_environment(cfg.config_hash)
        if not args.quick: validate_full_hardware(environment)
        output = Path(cfg.paths.output_dir); atomic_write_json(output / "environment.json", environment)
        print(json.dumps(environment, indent=2)); return 0
    if args.command == "report":
        from .reporting.report import write_report
        path = write_report(cfg.paths.output_dir, _load_evidence(Path(cfg.paths.output_dir)))
        print(path); return 0
    from .pipeline import run_command
    return run_command(args.command, cfg, args)


def _load_evidence(output):
    path = output / "evidence.json"
    return json.loads(path.read_text()) if path.exists() else {}


if __name__ == "__main__":
    raise SystemExit(main())
