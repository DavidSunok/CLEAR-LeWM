#!/usr/bin/env python3
"""Attach a verified local checkpoint record to an existing result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clear_lewm.runner import _checkpoint_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    checkpoint = _checkpoint_record(args.policy_id, args.cache_dir)
    if checkpoint is None or "runtime_sha256" not in checkpoint:
        raise ValueError("Could not resolve a unique runtime checkpoint")
    source = checkpoint.get("source", {})
    if source.get("tensors_loaded") != 303:
        raise ValueError("Checkpoint source does not certify a 303-tensor load")
    result["checkpoint"] = checkpoint
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
