#!/usr/bin/env python3
"""Classify and compare two CLEAR result files episode by episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clear_lewm.reproduction import compare_reproduction_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare_reproduction_results(args.reference, args.candidate)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)

    failed = report["classification"] in {"incompatible", "same-runtime-drift"}
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
