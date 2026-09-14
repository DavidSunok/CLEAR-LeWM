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
        "01  /  SINGLE-TASK TRAINING",
        "Task-specific E1 checkpoints  /  four tasks evaluated separately",
    ),
    (
        ROOT / "results" / "v0.8" / "intact-unified-chart.json",
        ROOT / "assets" / "intact_unified_comparison_v08.png",
        "02  /  SHARED-ENCODER TRAINING",
        "Unified E5 checkpoint  /  shared encoder, four task heads",
    ),
)
TASKS = (
    ("pusht", "PushT"),
    ("cube", "Cube"),
    ("reacher", "Reacher"),
    ("tworoom", "TwoRoom"),
)
MODELS = (
    ("official-lewm", "Official LeWM", "CEM 300 x 30", "#E45D52"),
    ("dinov2-no-proprio-lewm", "DINOv2 no-proprio", "CEM 300 x 30", "#3778B5"),
    ("gcbc-joint-lewm", "GCBC Joint", "CEM 300 x 30", "#269477"),
    ("intact-direct", "INTACT", "Direct / no search", "#B77321"),
    ("intact-guarded-a", "INTACT", "Guarded A 128 x 3", "#8561B1"),
)
SIZE = (2048, 1180)
PANEL_TOP = 296
PANEL_WIDTH = 962
PANEL_HEIGHT = 796


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
    draw.text((x0 + 30, y0 + 20), mode.upper(), fill="#1C2B31", font=font(26, True))
    draw.text((x0 + 30, y0 + 56), "Success rate (%)", fill="#627276", font=font(17))
    draw.text((x0 + 752, y0 + 32), "RED = BEST SR", fill="#B4232B", font=font(15, True))

    plot_left = x0 + 111
    plot_right = x0 + 708
    plot_span = plot_right - plot_left
    value_x = x0 + 752
    group_top = y0 + 128
    group_height = 158

    for task_index in range(len(TASKS)):
        top = group_top + task_index * group_height
        if task_index % 2 == 1:
            draw.rectangle(
                (x0 + 18, top - 10, x0 + PANEL_WIDTH - 18, top + 146),
                fill="#F5F8F7",
            )
        if task_index:
            draw.line(
                (x0 + 28, top - 11, x0 + PANEL_WIDTH - 28, top - 11),
                fill="#E4EBE9",
                width=2,
            )

    for tick in (0, 25, 50, 75, 100):
        x = plot_left + round(plot_span * tick / 100)
        label = str(tick)
        draw.line((x, y0 + 111, x, y0 + 745), fill="#E1E8E6", width=2)
        draw.text(
            (x - draw.textlength(label, font=font(15)) / 2, y0 + 90),
            label,
            fill="#66767B",
            font=font(15),
        )

    for task_index, (task, label) in enumerate(TASKS):
        top = group_top + task_index * group_height
        best = max(
            float(rows[(model, task, mode)]["success_rate_mean_percent"])
            for model, *_ in MODELS
            if (model, task, mode) in rows
        )
        draw.text((x0 + 30, top), label, fill="#172930", font=font(21, True))
        for model_index, (model, _, _, color) in enumerate(MODELS):
            row_y = top + 42 + model_index * 22
            row = rows.get((model, task, mode))
            if row is None:
                draw.text((value_x, row_y - 10), "n/a", fill="#94A3A2", font=font(16))
                continue

            mean = float(row["success_rate_mean_percent"])
            deviation = float(row["success_rate_sample_std_percent"])
            center = plot_left + round(plot_span * mean / 100)
            lower = plot_left + round(plot_span * max(0, mean - deviation) / 100)
            upper = plot_left + round(plot_span * min(100, mean + deviation) / 100)
            draw.line((plot_left, row_y, center, row_y), fill=color, width=7)
            draw.line((lower, row_y, upper, row_y), fill="#24343E", width=2)
            draw.line((lower, row_y - 5, lower, row_y + 5), fill="#24343E", width=2)
            draw.line((upper, row_y - 5, upper, row_y + 5), fill="#24343E", width=2)
            draw.ellipse(
                (center - 6, row_y - 6, center + 6, row_y + 6),
                fill=color,
                outline="#FFFFFF",
                width=2,
            )
            color_value = "#B4232B" if abs(mean - best) < 1e-8 else "#1A2C33"
            draw.text(
                (value_x, row_y - 11),
                f"{mean:.1f}",
                fill=color_value,
                font=font(18, True),
            )


def render(source: Path, output: Path, heading: str, subtitle: str) -> None:
    rows = load_rows(source)
    canvas = Image.new("RGB", SIZE, "#F2F6F5")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, SIZE[0], 158), fill="#121D29")
    draw.text((52, 25), heading, fill="#91E1D9", font=font(19, True))
    draw.text(
        (52, 62),
        "Five model-policy variants. One evaluation protocol.",
        fill="#FFFFFF",
        font=font(39, True),
    )
    draw.text((52, 121), subtitle, fill="#C6D2D7", font=font(18))

    draw.rectangle((0, 158, SIZE[0], 259), fill="#E9F0EE")
    for x, (_, label, method, color) in zip((52, 393, 797, 1148, 1520), MODELS):
        draw.line((x, 199, x + 23, 199), fill=color, width=8)
        draw.ellipse((x + 6, 192, x + 20, 206), fill=color, outline="#FFFFFF", width=2)
        draw.text((x + 35, 175), label, fill="#172930", font=font(19, True))
        draw.text((x + 35, 206), method, fill="#56686B", font=font(16))

    draw_panel(draw, rows, 52, "moderate")
    draw_panel(draw, rows, 1034, "strict")
    draw.text(
        (52, 1110),
        "CEM references are single-task and not epoch-matched; "
        "DINOv2 / GCBC have no Reacher checkpoint.",
        fill="#455B5E",
        font=font(17),
    )
    draw.text(
        (52, 1142),
        "Whiskers: s.d. across 3 eval seeds (CEM) or 3 training-seed means, "
        "each pooled over 3 eval seeds (INTACT).",
        fill="#607276",
        font=font(16),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(output)


def main() -> int:
    for source, output, heading, subtitle in FIGURES:
        render(source, output, heading, subtitle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
