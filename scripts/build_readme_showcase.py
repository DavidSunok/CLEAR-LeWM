#!/usr/bin/env python3
"""Build the branded README hero and headline benchmark graphic."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results" / "v0.3"
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
    return {
        task: Image.open(ASSETS / "task_gifs" / f"{task}.gif").convert("RGB")
        for task in TASKS
    }


def _draw_hero() -> None:
    width, height = 1600, 600
    canvas = Image.new("RGB", (width, height), "#0B1220")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 14, height), fill="#37B5A5")
    draw.rectangle((14, 0, 20, height), fill="#F97066")

    draw.text(
        (74, 58),
        "AUDITABLE ROBOT WORLD-MODEL EVALUATION",
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
        "Fixed pairs. Explicit completion. Verifiable runtimes.",
        fill="#FFFFFF",
        font=_font(29, bold=True),
    )

    metrics = (
        ("4", "benchmark tasks"),
        ("2", "auditable modes"),
        ("16", "v0.3 audited runs"),
        ("303/303", "official tensors"),
    )
    for index, (value, label) in enumerate(metrics):
        x = 74 + index * 214
        if index:
            draw.line((x - 22, 358, x - 22, 442), fill="#344054", width=2)
        value_font = _font(36 if index == 3 else 43, bold=True)
        draw.text((x, 354), value, fill="#F4C95D", font=value_font)
        draw.text((x, 410), label, fill="#98A2B3", font=_font(16))

    draw.text(
        (74, 517),
        "PAIR-LOCKED  /  RANDOM-CONTROLLED  /  RUNTIME-HASHED",
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


def _legend(draw, x, y, entries, text_color="#344054"):
    cursor = x
    for label, color in entries:
        draw.rectangle((cursor, y + 3, cursor + 18, y + 21), fill=color)
        draw.text((cursor + 27, y), label, fill=text_color, font=_font(16))
        cursor += int(draw.textlength(label, font=_font(16))) + 72


def _draw_results() -> None:
    canvas = Image.new("RGB", (1600, 700), "#F3F6FA")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1600, 126), fill="#0B1220")
    draw.text(
        (54, 28),
        "WHAT CHANGES WHEN SUCCESS MEANS COMPLETION?",
        fill="#FFFFFF",
        font=_font(31, bold=True),
    )
    draw.text(
        (56, 76),
        "Fixed 100-pair manifests per tier. Same 300 x 30 planner. Stricter "
        "protocols expose the floor while retaining capability.",
        fill="#B8C1D1",
        font=_font(18),
    )
    _legend(
        draw,
        1082,
        48,
        (
            ("Moderate random", "#98A2B3"),
            ("Strict random", "#37B5A5"),
            ("Strict LeWM", "#F97066"),
        ),
        text_color="#D0D5DD",
    )

    official_random = [_success_rate(task, "moderate", "random") for task in TASKS]
    strict_random = [_success_rate(task, "strict", "random") for task in TASKS]
    strict_model = [_success_rate(task, "strict", "official-lewm") for task in TASKS]
    images = _task_images()
    panel_width, panel_height = 360, 510
    for index, task in enumerate(TASKS):
        x = 52 + index * 378
        y = 146
        _rounded(
            draw,
            (x, y, x + panel_width, y + panel_height),
            "#FFFFFF",
            outline="#D8DEE9",
            width=2,
            radius=8,
        )
        draw.rectangle((x, y, x + panel_width, y + 7), fill=TASK_COLORS[task])
        draw.text(
            (x + 18, y + 18),
            TASK_LABELS[task],
            fill="#101828",
            font=_font(22, bold=True),
        )
        frame = ImageOps.fit(images[task], (324, 154), method=Image.Resampling.LANCZOS)
        canvas.paste(frame, (x + 18, y + 55))

        official = official_random[index]
        strict = strict_random[index]
        model = strict_model[index]
        reduction = official - strict
        gain = model - strict

        draw.text(
            (x + 18, y + 225),
            "RANDOM-POLICY FLOOR",
            fill="#667085",
            font=_font(13, bold=True),
        )
        draw.text(
            (x + 18, y + 251),
            f"{official:.0f}%",
            fill="#667085",
            font=_font(38, bold=True),
        )
        draw.text((x + 20, y + 299), "MODERATE", fill="#667085", font=_font(12, True))
        draw.line((x + 113, y + 276, x + 220, y + 276), fill="#B7C0CE", width=3)
        draw.polygon(
            ((x + 220, y + 270), (x + 232, y + 276), (x + 220, y + 282)),
            fill="#37B5A5",
        )
        draw.text(
            (x + 250, y + 251),
            f"{strict:.0f}%",
            fill="#159A8C",
            font=_font(38, bold=True),
        )
        draw.text((x + 255, y + 299), "STRICT", fill="#159A8C", font=_font(12, True))

        _rounded(
            draw,
            (x + 106, y + 314, x + 254, y + 348),
            "#E9F8F5",
            radius=6,
        )
        badge = f"-{reduction:.0f} pp floor"
        badge_width = draw.textlength(badge, font=_font(14, True))
        draw.text(
            (x + 180 - badge_width / 2, y + 322),
            badge,
            fill="#0E776D",
            font=_font(14, True),
        )

        draw.line((x + 18, y + 367, x + 342, y + 367), fill="#E4E7EC", width=2)
        draw.text(
            (x + 18, y + 383),
            "STRICT, SAME PAIRS",
            fill="#667085",
            font=_font(13, bold=True),
        )
        draw.text(
            (x + 18, y + 410),
            f"LeWM  {model:.0f}%",
            fill="#E4514B",
            font=_font(24, bold=True),
        )
        draw.text(
            (x + 203, y + 415),
            f"Random  {strict:.0f}%",
            fill="#667085",
            font=_font(16, bold=True),
        )
        draw.text(
            (x + 18, y + 465),
            f"+{gain:.0f} pp capability above random",
            fill="#344054",
            font=_font(14, bold=True),
        )

    draw.text(
        (54, 674),
        "Official LeWM checkpoints  |  seed 42  |  per-task fixed manifests  |  "
        "raw success rate",
        fill="#667085",
        font=_font(13),
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
