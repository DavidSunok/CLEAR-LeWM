#!/usr/bin/env python3
"""Fingerprint the seeded CEM noise stream and a representative CUDA GEMM."""

from __future__ import annotations

import argparse
import hashlib
import json


def _tensor_sha256(tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cem-rounds", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--matmul-precision", default="highest")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA accelerator is required")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    noise_hashes = []
    for _ in range(args.cem_rounds):
        noise = torch.randn(
            1,
            args.num_samples,
            args.horizon,
            args.action_dim,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        noise_hashes.append(_tensor_sha256(noise))

    torch.manual_seed(20260827)
    left = torch.randn(300, 768, dtype=torch.float32)
    right = torch.randn(768, 3072, dtype=torch.float32)
    torch.set_float32_matmul_precision(args.matmul_precision)
    product = left.cuda() @ right.cuda()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "schema_version": "clear-lewm-cross-accelerator-numerics-v1",
        "accelerator": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
        },
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cem_noise": {
            "seed": args.seed,
            "rounds": args.cem_rounds,
            "num_samples": args.num_samples,
            "horizon": args.horizon,
            "action_dim": args.action_dim,
            "round_hashes_sha256": hashlib.sha256(
                "".join(noise_hashes).encode()
            ).hexdigest(),
        },
        "representative_fp32_gemm": {
            "shape": [[300, 768], [768, 3072]],
            "sha256": _tensor_sha256(product),
            "sum_float64": float(product.double().sum().cpu()),
            "absmax": float(product.abs().max().cpu()),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
