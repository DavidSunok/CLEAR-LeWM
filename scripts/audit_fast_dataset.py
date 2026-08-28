#!/usr/bin/env python3
"""Check a FAST memory map against its authoritative source reader."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from clear_lewm.fast_dataset import FastMemmapDataset
from clear_lewm.fast_profiles import (
    FAST_TASK_PROFILES,
    get_fast_profile,
    load_profile_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source")
    source_group.add_argument("--task", choices=sorted(FAST_TASK_PROFILES))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--fast-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--num-clips", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output")
    args = parser.parse_args()

    import stable_worldmodel as swm

    kwargs = {"num_steps": args.num_steps, "frameskip": args.frameskip}
    source = (
        load_profile_dataset(args.task, args.cache_dir, transform=None, **kwargs)
        if args.task
        else swm.data.load_dataset(
            args.source, transform=None, cache_dir=args.cache_dir, **kwargs
        )
    )
    fast = FastMemmapDataset(args.fast_dir, transform=None, **kwargs)
    if args.task:
        profile = get_fast_profile(args.task)
        if fast.meta.get("task") != args.task:
            raise ValueError(
                f"FAST task mismatch: expected {args.task!r}, "
                f"got {fast.meta.get('task')!r}"
            )
        if fast.meta.get("source") != profile.source:
            raise ValueError(
                f"FAST source mismatch: expected {profile.source!r}, "
                f"got {fast.meta.get('source')!r}"
            )
    report = {
        "source": fast.meta["source"],
        "task": args.task,
        "length_equal": len(source) == len(fast),
        "episode_lengths_equal": bool(np.array_equal(source.lengths, fast.lengths)),
        "episode_offsets_equal": bool(np.array_equal(source.offsets, fast.offsets)),
        "columns": {},
        "clips_checked": 0,
    }
    for name in fast.column_names:
        if name == "pixels":
            report["columns"][name] = {
                "scope": "seeded-clips",
                "exact": True,
                "mae": 0.0,
                "max_abs_error": 0.0,
            }
            continue
        left = np.asarray(source.get_col_data(name))
        right = np.asarray(fast.get_col_data(name))
        error = np.abs(left.astype(np.float64) - right.astype(np.float64))
        report["columns"][name] = {
            "shape_equal": left.shape == right.shape,
            "exact": bool(np.array_equal(left, right, equal_nan=True)),
            "mae": float(np.nanmean(error)),
            "max_abs_error": float(np.nanmax(error)),
        }

    rng = np.random.default_rng(args.seed)
    indices = np.unique(
        np.concatenate(([0, len(fast) - 1], rng.integers(0, len(fast), args.num_clips)))
    )
    pixel_error_sum = 0.0
    pixel_values = 0
    for index in indices:
        left, right = source[int(index)], fast[int(index)]
        for name in fast.column_names:
            left_tensor = torch.as_tensor(left[name])
            right_tensor = torch.as_tensor(right[name])
            difference = (left_tensor.float() - right_tensor.float()).abs()
            if name == "pixels":
                pixel_error_sum += float(difference.sum())
                pixel_values += difference.numel()
                report["columns"][name]["max_abs_error"] = max(
                    report["columns"][name]["max_abs_error"],
                    float(difference.max()),
                )
                if not torch.equal(left_tensor, right_tensor):
                    report["columns"][name]["exact"] = False
            torch.testing.assert_close(
                left_tensor, right_tensor, rtol=0, atol=0, equal_nan=True
            )
    if "pixels" in report["columns"]:
        report["columns"]["pixels"]["mae"] = pixel_error_sum / max(pixel_values, 1)
    report["clips_checked"] = int(len(indices))
    report["pass"] = bool(
        report["length_equal"]
        and report["episode_lengths_equal"]
        and report["episode_offsets_equal"]
        and all(item["exact"] for item in report["columns"].values())
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(payload)
    print(payload, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
