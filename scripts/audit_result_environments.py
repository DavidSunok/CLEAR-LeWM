#!/usr/bin/env python3
"""Group CLEAR results by physics, numerical, and execution fingerprints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def result_files(inputs: list[Path]) -> list[Path]:
    files = set()
    for item in inputs:
        if item.is_file():
            files.add(item.resolve())
        elif item.is_dir():
            files.update(path.resolve() for path in item.rglob("*.json"))
        else:
            raise FileNotFoundError(item)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-physics",
        action="store_true",
        help="fail unless every result has one common physics fingerprint",
    )
    parser.add_argument(
        "--strict-numerics",
        action="store_true",
        help="fail unless every result has one common numerical fingerprint",
    )
    parser.add_argument(
        "--strict-execution",
        action="store_true",
        help="fail unless every result has one common execution fingerprint",
    )
    args = parser.parse_args()

    groups = {"physics": defaultdict(lambda: defaultdict(list))}
    groups["numerics"] = defaultdict(lambda: defaultdict(list))
    groups["execution"] = defaultdict(lambda: defaultdict(list))
    missing = []
    missing_execution = []
    parsed = 0
    for path in result_files(args.inputs):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") != "clear-lewm-result-v1":
            continue
        parsed += 1
        environment = payload.get("environment")
        if not environment:
            missing.append(str(path))
            continue
        task = payload.get("task", "unknown")
        groups["physics"][task][environment["physics_fingerprint"]].append(str(path))
        groups["numerics"][task][environment["numerics_fingerprint"]].append(str(path))
        execution_fingerprint = environment.get("execution_fingerprint")
        if execution_fingerprint is None:
            missing_execution.append(str(path))
        else:
            groups["execution"][task][execution_fingerprint].append(str(path))

    def sorted_groups(kind: str) -> dict:
        return {
            task: dict(sorted(fingerprints.items()))
            for task, fingerprints in sorted(groups[kind].items())
        }

    report = {
        "schema_version": "clear-lewm-environment-audit-v1",
        "results": parsed,
        "results_without_environment": missing,
        "results_without_execution_fingerprint": missing_execution,
        "physics_groups_by_task": sorted_groups("physics"),
        "numerics_groups_by_task": sorted_groups("numerics"),
        "execution_groups_by_task": sorted_groups("execution"),
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")

    physics_failed = args.strict_physics and (
        missing or any(len(items) != 1 for items in groups["physics"].values())
    )
    numerics_failed = args.strict_numerics and (
        missing or any(len(items) != 1 for items in groups["numerics"].values())
    )
    execution_failed = args.strict_execution and (
        missing
        or missing_execution
        or any(len(items) != 1 for items in groups["execution"].values())
    )
    return 1 if physics_failed or numerics_failed or execution_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
