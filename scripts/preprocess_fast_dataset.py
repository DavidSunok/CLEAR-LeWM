#!/usr/bin/env python3
"""Convert a stable-worldmodel dataset to an auditable raw memory map."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _write_meta(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_resume(output: Path, expected: dict) -> None:
    meta_path = output / "meta.json"
    if not meta_path.exists():
        raise ValueError("--resume requires an existing incomplete FAST snapshot")
    existing = json.loads(meta_path.read_text())
    if existing.get("complete", False):
        raise ValueError("Refusing to resume a FAST snapshot already marked complete")
    for key in ("schema_version", "source", "rows", "shapes", "dtypes"):
        if existing.get(key) != expected[key]:
            raise ValueError(f"FAST resume metadata mismatch for {key!r}")
    for name, shape in expected["shapes"].items():
        path = output / f"{name}.bin"
        required_bytes = (
            int(np.prod(shape)) * np.dtype(expected["dtypes"][name]).itemsize
        )
        if not path.exists() or path.stat().st_size != required_bytes:
            raise ValueError(f"FAST resume file is missing or malformed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--resume", type=int, default=0)
    args = parser.parse_args()

    import stable_worldmodel as swm
    import torch

    dataset = swm.data.load_dataset(args.source, cache_dir=args.cache_dir)
    rows = len(dataset)
    if args.resume < 0 or args.resume > rows:
        raise ValueError(f"--resume must be between 0 and {rows}")
    first = dataset[0]

    def row_array(value):
        array = torch.as_tensor(value).detach().cpu().numpy()
        return array[0] if array.shape[:1] == (1,) else array

    columns = {key: row_array(value) for key, value in first.items()}
    shapes = {name: [rows, *value.shape] for name, value in columns.items()}
    dtypes = {name: str(value.dtype) for name, value in columns.items()}

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "clear-fast-memmap-v1",
        "complete": False,
        "source": args.source,
        "rows": rows,
        "episodes": int(len(dataset.lengths)),
        "shapes": shapes,
        "dtypes": dtypes,
    }
    if args.resume:
        _validate_resume(args.output, metadata)
        saved_lengths = np.load(args.output / "episode_lengths.npy")
        saved_offsets = np.load(args.output / "episode_offsets.npy")
        if not np.array_equal(saved_lengths, np.asarray(dataset.lengths)):
            raise ValueError("FAST resume episode lengths do not match the source")
        if not np.array_equal(saved_offsets, np.asarray(dataset.offsets)):
            raise ValueError("FAST resume episode offsets do not match the source")
    else:
        _write_meta(args.output / "meta.json", metadata)
        np.save(args.output / "episode_lengths.npy", np.asarray(dataset.lengths))
        np.save(args.output / "episode_offsets.npy", np.asarray(dataset.offsets))

    mode = "r+" if args.resume else "w+"
    arrays = {
        name: np.memmap(
            args.output / f"{name}.bin",
            dtype=np.dtype(dtypes[name]),
            mode=mode,
            shape=tuple(shape),
        )
        for name, shape in shapes.items()
    }
    started = time.monotonic()
    for start in range(args.resume, rows, args.batch_size):
        end = min(start + args.batch_size, rows)
        batch = dataset.__getitems__(list(range(start, end)))
        for offset, sample in enumerate(batch):
            for name, array in arrays.items():
                array[start + offset] = row_array(sample[name])
        if end == rows or start % (args.batch_size * 50) == 0:
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = (end - args.resume) / elapsed
            print(f"{end}/{rows} ({100 * end / rows:.1f}%), {rate:.0f} rows/s")
    for array in arrays.values():
        array.flush()
    metadata["complete"] = True
    _write_meta(args.output / "meta.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
