#!/usr/bin/env python3
"""Render single-task and shared-encoder v0.8 checkpoint comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "v0.8" / "summary.json"
FIGURES = (
    (
        ROOT / "results" / "v0.8" / "intact-e1-chart.json",
        ROOT / "assets" / "community_model_comparison_v08.png",
        "01 / SINGLE-TASK E1",
        "Each task trained separately",
    ),
    (
        ROOT / "results" / "v0.8" / "intact-unified-chart.json",
        ROOT / "assets" / "intact_unified_comparison_v08.png",
        "02 / UNIFIED E5",
        "Four tasks, one encoder",
    ),
)
TASKS = (
    ("pusht", "PushT"),
    ("cube", "Cube"),
    ("reacher", "Reacher"),
    ("tworoom", "TwoRoom"),
)
MODELS = (
    ("official-lewm", "LeWM / CEM", "#E45D52"),
    ("dinov2-no-proprio-lewm", "DINOv2 / CEM", "#3778B5"),
    ("gcbc-joint-lewm", "GCBC / CEM", "#269477"),
    ("intact-direct", "INTACT / Direct", "#B77321"),
    ("intact-guarded-a", "INTACT / Guarded A", "#8561B1"),
)
SIZE = (2048, 1110)
PANEL_TOP = 240
PANEL_WIDTH = 962
PANEL_HEIGHT = 824


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path("/usr/share/fonts/truetype/dejavu") / name), size=size
    )


def load_rows(source: Path) -> dict[tuple[str, str, str], dict]:
    reference = json.loads(SUMMARY.read_text())["rows"]
    intact = json.loads(source.read_text())["rows"]
    rows = {(r["model"], r["task"], r["protocol"]): r for r in reference + intact}
    for model, *_ in MODELS:
        for task, _ in TASKS:
            for mode in ("moderate", "strict"):
                if task == "reacher" and model in (
                    "dinov2-no-proprio-lewm",
                    "gcbc-joint-lewm",
                ):
                    continue
                if (model, task, mode) not in rows:
                    raise ValueError(f"Missing comparison row: {model}, {task}, {mode}")
    return rows


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rows: dict[tuple[str, str, str], dict],
    x0: int,
    mode: str,
) -> None:
    y0 = PANEL_TOP
    draw.rounded_rectangle(
        (x0, y0, x0 + PANEL_WIDTH, y0 + PANEL_HEIGHT),
        radius=6,
        fill="#FFFFFF",
        outline="#D8E0DF",
        width=2,
    )
    draw.text(
        (x0 + 30, y0 + 20),
        f"{mode.upper()}  |  SR (%)",
        fill="#1C2B31",
        font=font(30, True),
    )
    draw.text((x0 + 752, y0 + 25), "BEST SR", fill="#B4232B", font=font(20, True))

    plot_left = x0 + 111
    plot_right = x0 + 708
    plot_span = plot_right - plot_left
    value_x = x0 + 752
    group_top = y0 + 130
    group_height = 170

    for task_index in range(len(TASKS)):
        top = group_top + task_index * group_height
        if task_index % 2 == 1:
            draw.rectangle(
                (x0 + 18, top - 10, x0 + PANEL_WIDTH - 18, top + 160),
                fill="#F5F8F7",
            )
        if task_index:
            draw.line(
                (x0 + 28, top - 11, x0 + PANEL_WIDTH - 28, top - 11),
                fill="#E4EBE9",
                width=2,
            )

    for tick in (0, 50, 100):
        x = plot_left + round(plot_span * tick / 100)
        label = str(tick)
        for task_index in range(len(TASKS)):
            top = group_top + task_index * group_height
            draw.line((x, top + 36, x, top + 152), fill="#E1E8E6", width=2)
        draw.text(
            (x - draw.textlength(label, font=font(20)) / 2, y0 + 86),
            label,
            fill="#66767B",
            font=font(20),
        )

    for task_index, (task, label) in enumerate(TASKS):
        top = group_top + task_index * group_height
        best = max(
            float(rows[(model, task, mode)]["success_rate_mean_percent"])
            for model, *_ in MODELS
            if (model, task, mode) in rows
        )
        draw.text((x0 + 30, top), label, fill="#172930", font=font(26, True))
        for model_index, (model, _, color) in enumerate(MODELS):
            row_y = top + 44 + model_index * 27
            row = rows.get((model, task, mode))
            if row is None:
                draw.text((value_x, row_y - 14), "n/a", fill="#94A3A2", font=font(21))
                continue

            mean = float(row["success_rate_mean_percent"])
            deviation = float(row["success_rate_sample_std_percent"])
            center = plot_left + round(plot_span * mean / 100)
            lower = plot_left + round(plot_span * max(0, mean - deviation) / 100)
            upper = plot_left + round(plot_span * min(100, mean + deviation) / 100)
            draw.line((plot_left, row_y, center, row_y), fill=color, width=10)
            draw.line((lower, row_y, upper, row_y), fill="#24343E", width=2)
            draw.line((lower, row_y - 5, lower, row_y + 5), fill="#24343E", width=2)
            draw.line((upper, row_y - 5, upper, row_y + 5), fill="#24343E", width=2)
            draw.ellipse(
                (center - 8, row_y - 8, center + 8, row_y + 8),
                fill=color,
                outline="#FFFFFF",
                width=2,
            )
            color_value = "#B4232B" if abs(mean - best) < 1e-8 else "#1A2C33"
            draw.text(
                (value_x, row_y - 15),
                f"{mean:.1f}",
                fill=color_value,
                font=font(24, True),
            )


def render(source: Path, output: Path, heading: str, title: str) -> None:
    rows = load_rows(source)
    canvas = Image.new("RGB", SIZE, "#F2F6F5")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, SIZE[0], 128), fill="#121D29")
    draw.text((52, 18), heading, fill="#91E1D9", font=font(23, True))
    draw.text((52, 58), title, fill="#FFFFFF", font=font(45, True))

    draw.rectangle((0, 128, SIZE[0], 211), fill="#E9F0EE")
    for x, (_, label, color) in zip((52, 385, 787, 1148, 1518), MODELS):
        draw.line((x, 170, x + 29, 170), fill=color, width=10)
        draw.ellipse((x + 8, 162, x + 24, 178), fill=color, outline="#FFFFFF", width=2)
        draw.text((x + 42, 151), label, fill="#172930", font=font(23, True))

    draw_panel(draw, rows, 52, "moderate")
    draw_panel(draw, rows, 1034, "strict")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(output)


def main() -> int:
    for source, output, heading, title in FIGURES:
        render(source, output, heading, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
