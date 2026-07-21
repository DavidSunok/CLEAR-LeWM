from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_dataset
from .manifests import generate_manifest, save_manifest
from .metrics import load_success_trace, summarize_success
from .protocols import PROTOCOLS, TASKS
from .runner import evaluate_manifest


def _write_or_print(data: dict, output: str | None) -> None:
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    else:
        print(payload, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clear-lewm",
        description="Auditable evaluation protocols for LeWM-compatible models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("protocols", help="Print the registered protocol definitions")

    audit = commands.add_parser(
        "audit", help="Audit start-goal difficulty and trivial pairs"
    )
    audit.add_argument("dataset")
    audit.add_argument("--task", required=True, choices=TASKS)
    audit.add_argument("--goal-offset", type=int, default=25)
    audit.add_argument("--output")

    manifest = commands.add_parser(
        "manifest", help="Generate a fixed evaluation manifest"
    )
    manifest.add_argument("dataset")
    manifest.add_argument("--task", required=True, choices=TASKS)
    manifest.add_argument("--protocol", default="moderate", choices=PROTOCOLS)
    manifest.add_argument("--num-eval", type=int, default=100)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--split", choices=("all", "train", "heldout"))
    manifest.add_argument("--full-sha256", action="store_true")
    manifest.add_argument("--output", required=True)

    summarize = commands.add_parser("summarize", help="Summarize success traces")
    summarize.add_argument("results")
    summarize.add_argument("--random-results")
    summarize.add_argument("--hold-steps", type=int, default=1)
    summarize.add_argument("--bootstrap-samples", type=int, default=10_000)
    summarize.add_argument("--seed", type=int, default=0)
    summarize.add_argument("--output")

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a policy on a fixed manifest"
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--policy", default="random")
    evaluate.add_argument("--policy-label")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--cache-dir")
    evaluate.add_argument("--dataset-name")
    evaluate.add_argument("--dataset-path")
    evaluate.add_argument("--upstream-dir")
    evaluate.add_argument(
        "--runtime-dir",
        help="directory containing custom jepa.py/module.py implementations",
    )
    evaluate.add_argument("--policy-seed", type=int)
    evaluate.add_argument("--num-samples", type=int)
    evaluate.add_argument("--n-steps", type=int)
    evaluate.add_argument("--topk", type=int)
    evaluate.add_argument(
        "--solver-batch-size",
        type=int,
        help="CEM environments per GPU batch; default 1 reproduces upstream",
    )
    evaluate.add_argument(
        "--cpu-threads",
        type=int,
        help="PyTorch CPU threads; default 1 avoids oversubscription",
    )
    evaluate.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        help="PyTorch float32 matmul mode; unset preserves the runtime default",
    )
    evaluate.add_argument(
        "--strict-checkpoint",
        action="store_true",
        help="fail when checkpoint keys are missing or unexpected",
    )
    evaluate.add_argument("--random-results")
    evaluate.add_argument("--video-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "protocols":
        _write_or_print(
            {name: spec.to_dict() for name, spec in PROTOCOLS.items()}, None
        )
        return 0
    if args.command == "audit":
        report = audit_dataset(args.dataset, args.task, args.goal_offset)
        _write_or_print(report, args.output)
        return 0
    if args.command == "manifest":
        manifest = generate_manifest(
            args.dataset,
            task=args.task,
            protocol=args.protocol,
            num_eval=args.num_eval,
            seed=args.seed,
            split=args.split,
            full_sha256=args.full_sha256,
        )
        save_manifest(manifest, args.output)
        print(f"Wrote {len(manifest['pairs'])} pairs to {Path(args.output).resolve()}")
        return 0
    if args.command == "summarize":
        trace = load_success_trace(args.results)
        random_trace = (
            load_success_trace(args.random_results) if args.random_results else None
        )
        report = summarize_success(
            trace,
            random_trace=random_trace,
            hold_steps=args.hold_steps,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        _write_or_print(report, args.output)
        return 0
    if args.command == "evaluate":
        result = evaluate_manifest(
            manifest_path=args.manifest,
            policy=args.policy,
            output=args.output,
            cache_dir=args.cache_dir,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            upstream_dir=args.upstream_dir,
            runtime_dir=args.runtime_dir,
            policy_seed=args.policy_seed,
            num_samples=args.num_samples,
            n_steps=args.n_steps,
            topk=args.topk,
            random_results=args.random_results,
            video_dir=args.video_dir,
            policy_label=args.policy_label,
            solver_batch_size=args.solver_batch_size,
            cpu_threads=args.cpu_threads,
            matmul_precision=args.matmul_precision,
            strict_checkpoint=args.strict_checkpoint,
        )
        print(json.dumps(result["metrics"], indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
