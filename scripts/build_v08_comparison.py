#!/usr/bin/env python3
"""Render the audited v0.8 three-seed checkpoint comparison."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "v0.8" / "summary.json"
OUTPUT = ROOT / "assets" / "community_model_comparison_v08.png"
TASKS = ("pusht", "cube", "tworoom")
TASK_LABELS = {"pusht": "PushT", "cube": "Cube", "tworoom": "TwoRoom"}
MODELS = (
    ("official-lewm", "Official LeWM", "#E85F52"),
    ("dinov2-no-proprio-lewm", "DINOv2 No-Proprio", "#3478B8"),
    ("gcbc-joint-lewm", "GCBC Joint", "#2D9B78"),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path("/usr/share/fonts/truetype/dejavu") / name), size=size
    )


def load_rows() -> dict[tuple[str, str, str], dict]:
    payload = json.loads(SUMMARY.read_text())
    return {
        (row["model"], row["task"], row["protocol"]): row for row in payload["rows"]
    }


def error_bar(
    draw: ImageDraw.ImageDraw,
    x: float,
    mean: float,
    deviation: float,
    plot_top: int,
    plot_bottom: int,
) -> None:
    height = plot_bottom - plot_top

    def y(value: float) -> int:
        return plot_bottom - round(height * max(0.0, min(100.0, value)) / 100.0)

    upper = y(mean + deviation)
    lower = y(mean - deviation)
    draw.line((x, upper, x, lower), fill="#263442", width=2)
    draw.line((x - 7, upper, x + 7, upper), fill="#263442", width=2)
    draw.line((x - 7, lower, x + 7, lower), fill="#263442", width=2)


def panel(
    draw: ImageDraw.ImageDraw,
    rows: dict[tuple[str, str, str], dict],
    mode: str,
    x0: int,
) -> None:
    y0, width, height = 154, 730, 520
    x1, y1 = x0 + width, y0 + height
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=7,
        fill="#FFFFFF",
        outline="#CFD8D5",
        width=2,
    )
    draw.text((x0 + 28, y0 + 22), mode.upper(), fill="#17211F", font=font(20, True))
    draw.text(
        (x0 + 28, y0 + 52),
        "Mean success rate with sample s.d.",
        fill="#65736F",
        font=font(14),
    )

    plot_left = x0 + 66
    plot_right = x1 - 24
    plot_top = y0 + 112
    plot_bottom = y0 + 420
    plot_height = plot_bottom - plot_top
    for tick in (0, 25, 50, 75, 100):
        y = plot_bottom - round(plot_height * tick / 100)
        draw.line((plot_left, y, plot_right, y), fill="#E1E7E4", width=2)
        label = str(tick)
        label_width = draw.textlength(label, font=font(13))
        draw.text(
            (plot_left - label_width - 10, y - 8),
            label,
            fill="#71807C",
            font=font(13),
        )

    group_width = (plot_right - plot_left) / len(TASKS)
    bar_width, gap = 42, 10
    cluster_width = len(MODELS) * bar_width + (len(MODELS) - 1) * gap
    for task_index, task in enumerate(TASKS):
        center = plot_left + group_width * (task_index + 0.5)
        cluster_left = center - cluster_width / 2
        for model_index, (model, _, color) in enumerate(MODELS):
            row = rows[(model, task, mode)]
            mean = float(row["success_rate_mean_percent"])
            deviation = float(row["success_rate_sample_std_percent"])
            left = round(cluster_left + model_index * (bar_width + gap))
            right = left + bar_width
            top = plot_bottom - round(plot_height * mean / 100)
            draw.rounded_rectangle(
                (left, top, right, plot_bottom), radius=3, fill=color
            )
            error_bar(
                draw,
                left + bar_width / 2,
                mean,
                deviation,
                plot_top,
                plot_bottom,
            )
            value = f"{mean:.1f}"
            value_width = draw.textlength(value, font=font(13, True))
            label_y = max(plot_top - 4, top - 25)
            draw.text(
                (left + (bar_width - value_width) / 2, label_y),
                value,
                fill="#263442",
                font=font(13, True),
            )
        task_label = TASK_LABELS[task]
        label_width = draw.textlength(task_label, font=font(15, True))
        draw.text(
            (center - label_width / 2, plot_bottom + 18),
            task_label,
            fill="#34423F",
            font=font(15, True),
        )


def main() -> int:
    rows = load_rows()
    canvas = Image.new("RGB", (1600, 740), "#F2F5F3")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1600, 120), fill="#0B1220")
    draw.text(
        (52, 23),
        "MATCHED v0.8 CHECKPOINT AUDIT",
        fill="#80E1D3",
        font=font(18, True),
    )
    draw.text(
        (52, 53),
        "Three models. One RTX 4090 evaluation stack.",
        fill="#FFFFFF",
        font=font(31, True),
    )

    legend_x = 950
    for index, (_, label, color) in enumerate(MODELS):
        x = legend_x + (index % 2) * 300
        y = 23 + (index // 2) * 34
        draw.rectangle((x, y + 4, x + 18, y + 22), fill=color)
        draw.text((x + 27, y), label, fill="#D9E0E8", font=font(14, True))

    panel(draw, rows, "moderate", 50)
    panel(draw, rows, "strict", 820)
    draw.text(
        (52, 706),
        "Seeds 0, 1, 42  |  100 episodes each  |  pure CEM 300 x 30  |  "
        "actor warm-start off  |  shared released tasks",
        fill="#5F6D69",
        font=font(14),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, optimize=True)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
