"""Build the public community leaderboard from validated submission bundles."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from clear_lewm.submissions import validate_submission

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "submissions" / "leaderboard.json"
README_PATH = ROOT / "README.md"
CHART_PATH = ROOT / "assets" / "community_model_comparison.png"
README_START = "<!-- community-leaderboard:start -->"
README_END = "<!-- community-leaderboard:end -->"
TASK_ORDER = ("pusht", "cube", "reacher", "tworoom")
TASK_LABELS = {
    "pusht": "PushT",
    "cube": "Cube",
    "reacher": "Reacher",
    "tworoom": "TwoRoom",
}
MODEL_COLORS = ("#E85F52", "#416FBD", "#8DB5DF")


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path("/usr/share/fonts/truetype/dejavu") / filename), size=size
    )


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


def _comparison_models(registry: dict) -> list[dict]:
    return [
        registry["matched_official_reference"],
        *registry["submissions"],
    ]


def _result_lookup(model: dict) -> dict[tuple[str, str], float]:
    return {
        (result["task"], result["protocol"]): result["success_rate_percent"]
        for result in model["results"]
    }


def render_chart(registry: dict, source_sha256: str) -> bytes:
    width, height = 1600, 740
    canvas = Image.new("RGB", (width, height), "#F2F5F3")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 112), fill="#0B1220")
    draw.text(
        (52, 24),
        "MATCHED MODEL COMPARISON",
        fill="#80E1D3",
        font=_font(18, True),
    )
    draw.text(
        (52, 55),
        "Three LeWM variants. One canonical evaluation.",
        fill="#FFFFFF",
        font=_font(31, True),
    )
    draw.text(
        (955, 34),
        "Seed 42  /  100 episodes  /  pure CEM 300 x 30",
        fill="#B8C1D1",
        font=_font(16),
    )

    models = _comparison_models(registry)
    labels = ("Official LeWM", "DINOv2 No-Proprio", "GCBC Joint")
    legend_x = 956
    for index, (label, color) in enumerate(zip(labels, MODEL_COLORS, strict=True)):
        x = legend_x + (index % 2) * 290
        y = 65 + (index // 2) * 26
        draw.rectangle((x, y + 3, x + 16, y + 19), fill=color)
        draw.text((x + 25, y), label, fill="#D9E0E8", font=_font(14, True))

    lookups = [_result_lookup(model) for model in models]
    shared_tasks = [
        task
        for task in TASK_ORDER
        if all((task, "moderate") in lookup for lookup in lookups)
        and all((task, "strict") in lookup for lookup in lookups)
    ]
    panel_y = 142
    panel_width = 730
    panel_height = 510
    for protocol_index, protocol in enumerate(("moderate", "strict")):
        panel_x = 50 + protocol_index * 770
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
            radius=7,
            fill="#FFFFFF",
            outline="#CBD5D1",
            width=2,
        )
        draw.text(
            (panel_x + 28, panel_y + 22),
            protocol.upper(),
            fill="#152421",
            font=_font(20, True),
        )
        plot_left = panel_x + 68
        plot_right = panel_x + panel_width - 26
        plot_top = panel_y + 82
        plot_bottom = panel_y + 410
        plot_height = plot_bottom - plot_top
        for tick in (0, 25, 50, 75, 100):
            y = plot_bottom - round(plot_height * tick / 100)
            draw.line((plot_left, y, plot_right, y), fill="#DFE5E2", width=2)
            tick_text = str(tick)
            text_width = draw.textlength(tick_text, font=_font(13))
            draw.text(
                (plot_left - text_width - 10, y - 8),
                tick_text,
                fill="#71807C",
                font=_font(13),
            )

        group_width = (plot_right - plot_left) / len(shared_tasks)
        bar_width = 38
        bar_gap = 10
        cluster_width = len(models) * bar_width + (len(models) - 1) * bar_gap
        for task_index, task in enumerate(shared_tasks):
            center = plot_left + group_width * (task_index + 0.5)
            cluster_left = center - cluster_width / 2
            for model_index, (lookup, color) in enumerate(
                zip(lookups, MODEL_COLORS, strict=True)
            ):
                value = float(lookup[(task, protocol)])
                x0 = round(cluster_left + model_index * (bar_width + bar_gap))
                x1 = x0 + bar_width
                y0 = plot_bottom - round(plot_height * value / 100)
                draw.rounded_rectangle(
                    (x0, y0, x1, plot_bottom),
                    radius=3,
                    fill=color,
                )
                value_text = f"{value:.0f}"
                value_width = draw.textlength(value_text, font=_font(14, True))
                draw.text(
                    (x0 + (bar_width - value_width) / 2, y0 - 24),
                    value_text,
                    fill="#34423F",
                    font=_font(14, True),
                )
            task_label = TASK_LABELS[task]
            label_width = draw.textlength(task_label, font=_font(16, True))
            draw.text(
                (center - label_width / 2, plot_bottom + 18),
                task_label,
                fill="#263531",
                font=_font(16, True),
            )

    draw.text(
        (52, 688),
        "Raw SR (%) on shared tasks only. Reacher is omitted because both "
        "community models have no canonical submission.",
        fill="#5C6967",
        font=_font(15),
    )
    draw.text(
        (52, 713),
        "Model/random and excess-over-random records remain available in the "
        "audited community tables.",
        fill="#5C6967",
        font=_font(15),
    )
    buffer = io.BytesIO()
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("clear_lewm_registry_sha256", source_sha256)
    canvas.save(buffer, format="PNG", optimize=True, pnginfo=png_info)
    return buffer.getvalue()


def _chart_is_current(path: Path, source_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            return (
                image.size == (1600, 740)
                and image.info.get("clear_lewm_registry_sha256") == source_sha256
            )
    except OSError:
        return False


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
    registry_sha256 = hashlib.sha256(registry_text.encode()).hexdigest()
    chart_bytes = render_chart(registry, registry_sha256)
    readme_text = _replace_readme_section(
        README_PATH.read_text(), render_readme(registry)
    )

    stale = []
    if not REGISTRY_PATH.is_file() or REGISTRY_PATH.read_text() != registry_text:
        stale.append(REGISTRY_PATH.relative_to(ROOT))
    if README_PATH.read_text() != readme_text:
        stale.append(README_PATH.relative_to(ROOT))
    if not _chart_is_current(CHART_PATH, registry_sha256):
        stale.append(CHART_PATH.relative_to(ROOT))

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
    CHART_PATH.write_bytes(chart_bytes)


if __name__ == "__main__":
    main()
