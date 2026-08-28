#!/usr/bin/env python3
"""Rebind a result after a manifest-only schema extension, with safety checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from clear_lewm.protocols import protocol_from_dict

TASK_THRESHOLD_KEYS = {
    "pusht": ("pusht_position_threshold", "pusht_angle_threshold_deg"),
    "cube": ("cube_position_threshold_m", "cube_orientation_threshold_deg"),
    "reacher": ("reacher_joint_threshold_rad",),
    "tworoom": ("tworoom_distance_threshold",),
}

PAIR_SEMANTIC_KEYS = (
    "sampling",
    "split",
    "heldout_fraction",
    "exclude_initial_success",
    "goal_offset",
    "eval_budget",
    "reproduce_upstream_off_by_one",
)


def _manifest_path(path: Path) -> str:
    parts = path.parts
    index = parts.index("manifests") if "manifests" in parts else len(parts) - 1
    return Path(*parts[index:]).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    manifest = json.loads(args.manifest.read_text())
    task = result["task"]
    if task != manifest["task"]:
        raise ValueError(f"Task mismatch: {task} != {manifest['task']}")
    if len(result["episode_successes"]) != len(manifest["pairs"]):
        raise ValueError("Result and manifest episode counts differ")
    if (
        result.get("dataset_fingerprint", manifest["dataset"]["fingerprint"])
        != (manifest["dataset"]["fingerprint"])
    ):
        raise ValueError("Dataset fingerprints differ")

    old_protocol = result["protocol"]
    new_protocol = manifest["protocol"]
    for key in PAIR_SEMANTIC_KEYS:
        if old_protocol[key] != new_protocol[key]:
            raise ValueError(f"Pair semantic {key!r} changed")
    old_minimum = old_protocol.get("min_difficulty", {}).get(task)
    new_minimum = new_protocol.get("min_difficulty", {}).get(task)
    if old_minimum != new_minimum:
        raise ValueError("Task difficulty filter changed")

    old_criterion = result["criterion"]
    for key in TASK_THRESHOLD_KEYS[task]:
        if old_criterion.get(key) != new_protocol.get(key):
            raise ValueError(f"Task criterion {key!r} changed")
    spec = protocol_from_dict(new_protocol)
    if old_criterion["sustained_steps"] != spec.hold_steps(task):
        raise ValueError("Task hold duration changed")

    result["protocol"] = new_protocol
    result["criterion"] = {
        **old_criterion,
        "sustained_steps": spec.hold_steps(task),
    }
    result["manifest"] = _manifest_path(args.manifest)
    result["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    output = args.output or args.result
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
