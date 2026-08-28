#!/usr/bin/env python3
"""Audit and benchmark the four canonical FAST training-input profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from clear_lewm.fast_profiles import FAST_TASK_PROFILES


def _snapshot_complete(path: Path) -> bool:
    meta_path = path / "meta.json"
    return meta_path.is_file() and json.loads(meta_path.read_text()).get(
        "complete", False
    )


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    tasks = tuple(FAST_TASK_PROFILES)
    while True:
        incomplete = [
            task for task in tasks if not _snapshot_complete(args.snapshot_root / task)
        ]
        if not incomplete:
            break
        if not args.wait:
            raise RuntimeError(f"Incomplete FAST snapshots: {incomplete}")
        print(f"Waiting for FAST snapshots: {', '.join(incomplete)}", flush=True)
        time.sleep(args.poll_seconds)

    audit_dir = args.output_root / "audits"
    benchmark_dir = args.output_root / "benchmarks"
    audit_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    if not args.skip_audit:
        for task in tasks:
            _run(
                [
                    sys.executable,
                    str(root / "scripts" / "audit_fast_dataset.py"),
                    "--task",
                    task,
                    "--cache-dir",
                    str(args.cache_dir),
                    "--fast-dir",
                    str(args.snapshot_root / task),
                    "--num-steps",
                    "4",
                    "--frameskip",
                    "5",
                    "--num-clips",
                    "256",
                    "--seed",
                    str(args.seed),
                    "--output",
                    str(audit_dir / f"{task}.json"),
                ]
            )

    for task in tasks:
        _run(
            [
                sys.executable,
                str(root / "scripts" / "benchmark_fast_dataset.py"),
                "--task",
                task,
                "--cache-dir",
                str(args.cache_dir),
                "--fast-dir",
                str(args.snapshot_root / task),
                "--num-steps",
                "4",
                "--frameskip",
                "5",
                "--batch-size",
                "32",
                "--num-workers",
                "0",
                "--warmup-batches",
                "3",
                "--batches",
                "30",
                "--rounds",
                str(args.rounds),
                "--priming-rounds",
                "1",
                "--seed",
                str(args.seed),
                "--output",
                str(benchmark_dir / f"{task}.json"),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
