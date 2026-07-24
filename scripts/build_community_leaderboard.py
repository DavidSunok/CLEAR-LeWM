"""Build the public community leaderboard from validated submission bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clear_lewm.submissions import validate_submission

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "submissions" / "leaderboard.json"
README_PATH = ROOT / "README.md"
README_START = "<!-- community-leaderboard:start -->"
README_END = "<!-- community-leaderboard:end -->"
TASK_ORDER = ("pusht", "cube", "reacher", "tworoom")
TASK_LABELS = {
    "pusht": "PushT",
    "cube": "Cube",
    "reacher": "Reacher",
    "tworoom": "TwoRoom",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _markdown_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _format_rate(value: float) -> str:
    return f"{value:.0f}%" if float(value).is_integer() else f"{value:.2f}%"


def _format_result(result: dict | None) -> str:
    if result is None:
        return "-"
    model = _format_rate(result["success_rate_percent"])
    random = _format_rate(result["random_success_rate_percent"])
    excess = result["excess_over_random_pp"]
    sign = "+" if excess > 0 else ""
    return f"{model} / {random} ({sign}{excess:.0f} pp)"


def _clean_number(value: float) -> int | float:
    rounded = round(float(value), 10)
    if abs(rounded) < 1e-10:
        return 0
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _official_reference(root: Path) -> dict:
    results = []
    for task in TASK_ORDER:
        for protocol in ("moderate", "strict"):
            path = (
                root
                / "results"
                / "v0.5"
                / f"{task}-{protocol}-official-lewm-seed42-n100.json"
            )
            payload = _read_json(path)
            metrics = payload["metrics"]
            results.append(
                {
                    "task": task,
                    "protocol": protocol,
                    "success_rate_percent": _clean_number(
                        metrics["success_rate_percent"]
                    ),
                    "random_success_rate_percent": _clean_number(
                        metrics["random_success_rate_percent"]
                    ),
                    "excess_over_random_pp": _clean_number(
                        metrics["excess_over_random_pp"]
                    ),
                }
            )
    return {
        "method": "Official LeWM",
        "policy_seed": 42,
        "episodes_per_result": 100,
        "results": results,
    }


def build_registry(root: Path = ROOT) -> dict:
    submissions = []
    submission_root = root / "submissions"
    for manifest_path in sorted(submission_root.glob("*/*/submission.json")):
        payload = _read_json(manifest_path)
        report = validate_submission(manifest_path, repo_root=root)
        bundle = manifest_path.parent.relative_to(root).as_posix()
        method = payload["method"]
        contact = payload["contact"]
        results = sorted(
            (
                {
                    **item,
                    "success_rate_percent": _clean_number(item["success_rate_percent"]),
                    "random_success_rate_percent": _clean_number(
                        item["random_success_rate_percent"]
                    ),
                    "excess_over_random_pp": _clean_number(
                        item["excess_over_random_pp"]
                    ),
                }
                for item in report["results"]
            ),
            key=lambda item: (
                TASK_ORDER.index(item["task"]),
                ("moderate", "strict").index(item["protocol"]),
            ),
        )
        submissions.append(
            {
                "method": method["name"],
                "method_repository": method["repository"],
                "method_revision": method["revision"],
                "contact_github": contact["github"],
                "benchmark_version": report["benchmark_version"],
                "training_data_track": report["training_data_track"],
                "verification": report["verification_requested"],
                "bundle": bundle,
                "method_card": f"{bundle}/METHOD_CARD.md",
                "complete_four_task_matrix": report["complete_four_task_matrix"],
                "missing_results": report["missing_results"],
                "results": results,
            }
        )

    return {
        "schema_version": "clear-lewm-community-leaderboard-v1",
        "benchmark_version": "v0.5",
        "canonical_policy_seed": 42,
        "episodes_per_result": 100,
        "matched_official_reference": _official_reference(root),
        "submission_count": len(submissions),
        "submissions": submissions,
    }


def render_registry(registry: dict) -> str:
    return json.dumps(registry, indent=2, sort_keys=True) + "\n"


def render_readme(registry: dict) -> str:
    lines = [
        README_START,
        "",
        "Canonical community entries use policy seed 42 and 100 episodes per",
        "task/mode. Values are **model / paired random (excess)**. CI validates",
        "the bundle structure, canonical manifests, trace arithmetic, provenance,",
        "and topology; the verification label states whether execution was",
        "independently reproduced.",
        "",
        "| Method | Task | Moderate | Strict | Verification |",
        "|---|---|---:|---:|---|",
    ]
    for submission in registry["submissions"]:
        by_key = {
            (result["task"], result["protocol"]): result
            for result in submission["results"]
        }
        method = _markdown_text(submission["method"])
        card = submission["method_card"]
        verification = _markdown_text(submission["verification"])
        contact = _markdown_text(submission["contact_github"])
        first = True
        for task in TASK_ORDER:
            moderate = by_key.get((task, "moderate"))
            strict = by_key.get((task, "strict"))
            method_cell = f"[{method}]({card})" if first else ""
            verify_cell = (
                f"{verification}; [@{contact}](https://github.com/{contact})"
                if first
                else ""
            )
            lines.append(
                f"| {method_cell} | {TASK_LABELS[task]} | "
                f"{_format_result(moderate)} | {_format_result(strict)} | "
                f"{verify_cell} |"
            )
            first = False
    lines.extend(
        [
            "",
            "A dash means that task/mode was not submitted. Supplementary evidence",
            "that does not match the canonical manifest/seed contract remains in each",
            "method card and is not mixed into this table.",
            "",
            README_END,
        ]
    )
    return "\n".join(lines)


def _replace_readme_section(readme: str, section: str) -> str:
    if README_START not in readme or README_END not in readme:
        raise RuntimeError("README community leaderboard markers are missing")
    before, remainder = readme.split(README_START, 1)
    _, after = remainder.split(README_END, 1)
    return before.rstrip() + "\n\n" + section + after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail when generated artifacts are stale"
    )
    args = parser.parse_args()

    registry = build_registry()
    registry_text = render_registry(registry)
    readme_text = _replace_readme_section(
        README_PATH.read_text(), render_readme(registry)
    )

    stale = []
    if not REGISTRY_PATH.is_file() or REGISTRY_PATH.read_text() != registry_text:
        stale.append(REGISTRY_PATH.relative_to(ROOT))
    if README_PATH.read_text() != readme_text:
        stale.append(README_PATH.relative_to(ROOT))

    if args.check:
        if stale:
            joined = ", ".join(str(path) for path in stale)
            raise SystemExit(
                f"Community leaderboard is stale: {joined}. Run "
                "python scripts/build_community_leaderboard.py"
            )
        return

    REGISTRY_PATH.write_text(registry_text)
    README_PATH.write_text(readme_text)


if __name__ == "__main__":
    main()
