#!/usr/bin/env python3
"""Benchmark audited FAST memmap reads against the authoritative dataset."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def _measure_backend(backend, args, queue) -> None:
    import stable_worldmodel as swm

    from clear_lewm.fast_dataset import FastMemmapDataset

    kwargs = {"num_steps": args.num_steps, "frameskip": args.frameskip}
    dataset = (
        swm.data.load_dataset(
            args.source, transform=None, cache_dir=args.cache_dir, **kwargs
        )
        if backend == "source"
        else FastMemmapDataset(args.fast_dir, transform=None, **kwargs)
    )
    count = (args.warmup_batches + args.batches) * args.batch_size
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(dataset), size=count, dtype=np.int64)
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=False,
    )
    iterator = iter(loader)
    for _ in range(args.warmup_batches):
        next(iterator)
    started = time.perf_counter()
    samples = 0
    for _ in range(args.batches):
        batch = next(iterator)
        samples += len(next(iter(batch.values())))
    elapsed = time.perf_counter() - started
    queue.put({"length": len(dataset), "rate": samples / elapsed})


def _measure(backend, args) -> dict:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_measure_backend, args=(backend, args, queue))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{backend} benchmark subprocess exited {process.exitcode}")
    return queue.get()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--fast-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rates = {"source": [], "fast": []}
    lengths = {}
    for round_index in range(args.rounds):
        order = ("source", "fast") if round_index % 2 == 0 else ("fast", "source")
        for name in order:
            measurement = _measure(name, args)
            lengths[name] = measurement["length"]
            rates[name].append(measurement["rate"])
    if lengths["source"] != lengths["fast"]:
        raise ValueError(
            f"Dataset length mismatch: {lengths['source']} != {lengths['fast']}"
        )

    source_rate = statistics.median(rates["source"])
    fast_rate = statistics.median(rates["fast"])
    report = {
        "scope": "loader-only; no model forward/backward or image transform",
        "source": args.source,
        "fast_dir": str(args.fast_dir),
        "host": platform.node(),
        "config": {
            "num_steps": args.num_steps,
            "frameskip": args.frameskip,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "warmup_batches": args.warmup_batches,
            "measured_batches": args.batches,
            "rounds": args.rounds,
            "seed": args.seed,
        },
        "samples_per_second": rates,
        "median_samples_per_second": {
            "source": source_rate,
            "fast": fast_rate,
        },
        "fast_speedup": fast_rate / source_rate,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
