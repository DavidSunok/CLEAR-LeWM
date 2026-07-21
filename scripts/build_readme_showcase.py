#!/usr/bin/env python3
"""Build the branded README hero and headline benchmark graphic."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results" / "reference"
TASKS = ("pusht", "cube", "reacher", "tworoom")
TASK_LABELS = {
    "pusht": "PushT",
    "cube": "Cube",
    "reacher": "Reacher",
    "tworoom": "TwoRoom",
}
TASK_COLORS = {
    "pusht": "#37B5A5",
    "cube": "#F97066",
    "reacher": "#F4C95D",
    "tworoom": "#79B86A",
}


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path("/usr/share/fonts/truetype/dejavu") / filename), size=size
    )


def _rounded(draw, box, fill, outline=None, width=1, radius=8):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _task_images() -> dict[str, Image.Image]:
    frame = Image.open(ASSETS / "protocols.gif").convert("RGB")
    images = {}
    for index, task in enumerate(TASKS):
        x = 17 + index * 210
        images[task] = frame.crop((x, 86, x + 184, 270))
    return images


def _draw_hero() -> None:
    width, height = 1600, 600
    canvas = Image.new("RGB", (width, height), "#0B1220")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 14, height), fill="#37B5A5")
    draw.rectangle((14, 0, 20, height), fill="#F97066")

    draw.text(
        (74, 58),
        "ROBOT WORLD-MODEL EVALUATION",
        fill="#80E1D3",
        font=_font(22, bold=True),
    )
    draw.text((70, 102), "CLEAR-LeWM", fill="#FFFFFF", font=_font(78, bold=True))
    draw.text(
        (74, 203),
        "A benchmark is only as credible as what it controls.",
        fill="#D0D5DD",
        font=_font(27),
    )
    draw.text(
        (74, 250),
        "Fixed goals. Explicit success. Verifiable checkpoints.",
        fill="#FFFFFF",
        font=_font(29, bold=True),
    )

    metrics = (
        ("3", "explicit protocol tiers"),
        ("24", "checked reference runs"),
        ("303/303", "checkpoint tensors loaded"),
    )
    for index, (value, label) in enumerate(metrics):
        x = 74 + index * 270
        if index:
            draw.line((x - 28, 358, x - 28, 442), fill="#344054", width=2)
        draw.text((x, 350), value, fill="#F4C95D", font=_font(43, bold=True))
        draw.text((x, 410), label, fill="#98A2B3", font=_font(17))

    draw.text(
        (74, 517),
        "OFFICIAL COMPATIBILITY  /  MODERATE ROBUSTNESS  /  STRICT COMPLETION",
        fill="#80E1D3",
        font=_font(18, bold=True),
    )

    images = _task_images()
    card_width, card_height = 262, 238
    for index, task in enumerate(TASKS):
        col, row = index % 2, index // 2
        x = 1015 + col * 280
        y = 48 + row * 254
        _rounded(draw, (x, y, x + card_width, y + card_height), "#FFFFFF", radius=6)
        draw.rectangle((x, y, x + 7, y + card_height), fill=TASK_COLORS[task])
        image = images[task].resize((226, 184))
        canvas.paste(image, (x + 22, y + 38))
        draw.text(
            (x + 22, y + 9),
            TASK_LABELS[task],
            fill="#101828",
            font=_font(18, bold=True),
        )

    output = ASSETS / "readme_hero.png"
    canvas.save(output, optimize=True)


def _success_rate(task: str, tier: str, policy: str) -> float:
    path = RESULTS / f"{task}-{tier}-{policy}-seed42-n100.json"
    return float(json.loads(path.read_text())["metrics"]["success_rate_percent"])


def _bar_group(draw, origin, values_a, values_b, labels, colors, maximum, size):
    x0, y0 = origin
    plot_width, plot_height = size
    draw.line((x0, y0, x0, y0 - plot_height), fill="#98A2B3", width=2)
    draw.line((x0, y0, x0 + plot_width, y0), fill="#98A2B3", width=2)
    for tick in range(0, maximum + 1, 10):
        y = y0 - int(plot_height * tick / maximum)
        draw.line((x0, y, x0 + plot_width, y), fill="#EAECF0", width=1)
        draw.text((x0 - 44, y - 10), str(tick), fill="#667085", font=_font(14))

    group_width = plot_width / len(labels)
    bar_width = 42
    for index, label in enumerate(labels):
        center = x0 + group_width * (index + 0.5)
        for offset, value, color in (
            (-bar_width - 3, values_a[index], colors[0]),
            (3, values_b[index], colors[1]),
        ):
            top = y0 - int(plot_height * value / maximum)
            visible_top = min(top, y0 - 2)
            draw.rectangle(
                (
                    int(center + offset),
                    visible_top,
                    int(center + offset + bar_width),
                    y0 - 1,
                ),
                fill=color,
            )
            draw.text(
                (int(center + offset), visible_top - 25),
                f"{value:.0f}",
                fill="#101828",
                font=_font(15, bold=True),
            )
        text_width = draw.textlength(label, font=_font(15, bold=True))
        draw.text(
            (int(center - text_width / 2), y0 + 12),
            label,
            fill="#344054",
            font=_font(15, bold=True),
        )


def _legend(draw, x, y, entries):
    cursor = x
    for label, color in entries:
        draw.rectangle((cursor, y + 3, cursor + 18, y + 21), fill=color)
        draw.text((cursor + 27, y), label, fill="#344054", font=_font(16))
        cursor += int(draw.textlength(label, font=_font(16))) + 72


def _draw_results() -> None:
    canvas = Image.new("RGB", (1600, 590), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (58, 34),
        "Stricter evaluation removes false confidence, not capable models.",
        fill="#101828",
        font=_font(34, bold=True),
    )
    draw.text(
        (60, 84),
        "The same fixed 100-pair manifests expose random-policy inflation "
        "while preserving meaningful model success.",
        fill="#475467",
        font=_font(19),
    )
    draw.line((800, 136, 800, 520), fill="#D0D5DD", width=2)

    labels = [TASK_LABELS[task] for task in TASKS]
    official_random = [_success_rate(task, "official", "random") for task in TASKS]
    strict_random = [_success_rate(task, "strict", "random") for task in TASKS]
    strict_model = [_success_rate(task, "strict", "official-lewm") for task in TASKS]

    draw.text((58, 138), "Random-policy floor", fill="#101828", font=_font(23, True))
    draw.text(
        (58, 170),
        "Upstream first-hit rules vs. strict task completion",
        fill="#667085",
        font=_font(16),
    )
    _legend(draw, 410, 144, (("Official", "#98A2B3"), ("Strict", "#37B5A5")))
    _bar_group(
        draw,
        (88, 490),
        official_random,
        strict_random,
        labels,
        ("#98A2B3", "#37B5A5"),
        maximum=50,
        size=(650, 270),
    )

    draw.text((844, 138), "Strict completion", fill="#101828", font=_font(23, True))
    draw.text(
        (844, 170),
        "Official LeWM remains decisively above random",
        fill="#667085",
        font=_font(16),
    )
    _legend(
        draw,
        1162,
        144,
        (("LeWM", "#F97066"), ("Random", "#D0D5DD")),
    )
    _bar_group(
        draw,
        (874, 490),
        strict_model,
        strict_random,
        labels,
        ("#F97066", "#D0D5DD"),
        maximum=50,
        size=(650, 270),
    )

    draw.text(
        (58, 558),
        "100 episodes  |  seed 42  |  fixed manifests  |  official LeWM "
        "checkpoints  |  CEM 300 x 30",
        fill="#667085",
        font=_font(15),
    )
    output = ASSETS / "headline_results.png"
    canvas.save(output, optimize=True)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    _draw_hero()
    _draw_results()
    print(ASSETS / "readme_hero.png")
    print(ASSETS / "headline_results.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
