#!/usr/bin/env python3
"""Build the branded README hero and headline benchmark graphic."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from statistics import geometric_mean

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results" / "v0.3"
BENCHMARKS = ROOT / "benchmarks"
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


@cache
def _fast_report(task: str) -> dict:
    path = BENCHMARKS / f"fast_loader_{task}_20260723.json"
    return json.loads(path.read_text())


def _fast_speedup(task: str) -> float:
    return float(_fast_report(task)["paired_speedup_median"])


def _fast_geomean() -> float:
    return geometric_mean(_fast_speedup(task) for task in TASKS)


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path("/usr/share/fonts/truetype/dejavu") / filename), size=size
    )


def _rounded(draw, box, fill, outline=None, width=1, radius=8):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _task_images() -> dict[str, Image.Image]:
    images = {}
    for task in TASKS:
        image = Image.open(ASSETS / "task_gifs" / f"{task}.gif")
        image.seek(image.n_frames // 2)
        images[task] = image.convert("RGB")
    return images


def _draw_hero() -> None:
    width, height = 1600, 600
    canvas = Image.new("RGB", (width, height), "#0B1220")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 664, height), fill="#25143D")
    draw.rectangle((0, 0, 15, height), fill="#B45CFF")
    draw.rectangle((15, 0, 21, height), fill="#37B5A5")

    draw.text(
        (70, 54),
        "AUDITABLE WORLD-MODEL EVALUATION",
        fill="#C995FF",
        font=_font(19, bold=True),
    )
    draw.text((66, 96), "CLEAR-LeWM", fill="#FFFFFF", font=_font(66, bold=True))
    draw.text(
        (70, 198),
        "TASK COMPLETION,",
        fill="#FFFFFF",
        font=_font(37, bold=True),
    )
    draw.text(
        (70, 246),
        "NOT PROXY SUCCESS.",
        fill="#C995FF",
        font=_font(37, bold=True),
    )
    draw.text(
        (72, 320),
        "Fixed goals. Explicit rules. Paired random floors.",
        fill="#C8D0DC",
        font=_font(20),
    )
    draw.line((70, 382, 622, 382), fill="#344054", width=2)
    facts = (("4", "TASKS"), ("2", "MODES"), ("16", "RUNS"))
    fact_value_font = _font(38, True)
    fact_label_font = _font(15, True)
    for index, (value, label) in enumerate(facts):
        x = 72 + index * 180
        draw.text((x, 408), value, fill="#80E1D3", font=fact_value_font)
        label_x = x + draw.textlength(value, font=fact_value_font) + 24
        draw.text((label_x, 426), label, fill="#98A2B3", font=fact_label_font)

    draw.rounded_rectangle(
        (70, 468, 622, 516),
        radius=5,
        fill="#32204B",
        outline="#76578F",
        width=1,
    )
    draw.text(
        (86, 475),
        f"{_fast_geomean():.1f}x",
        fill="#80E1D3",
        font=_font(27, True),
    )
    draw.text(
        (190, 474),
        "4-TASK GEOMETRIC-MEAN SPEEDUP",
        fill="#FFFFFF",
        font=_font(13, True),
    )
    draw.text(
        (190, 494),
        "paired steady-state  /  loader-only",
        fill="#B7A9C5",
        font=_font(12),
    )
    draw.text(
        (72, 536),
        "TARGET-ALIGNED  /  PHYSICS-VALID",
        fill="#C995FF",
        font=_font(13, bold=True),
    )
    draw.text(
        (72, 560),
        "NO TASK-IRRELEVANT STATE  /  NO PRE-SOLVED PAIRS",
        fill="#80E1D3",
        font=_font(13, bold=True),
    )

    draw.line((664, 46, 664, 554), fill="#344054", width=2)
    draw.text(
        (712, 48),
        "FROM LOOSE SIGNALS TO TASK CONTRACTS",
        fill="#FFFFFF",
        font=_font(29, True),
    )
    headers = (
        (712, "TASK"),
        (865, "RELEASED"),
        (1090, "CLEAR CONTRACT"),
        (1325, "STRICT SR"),
        (1478, "FAST"),
    )
    for x, label in headers:
        header_size = 17 if label in {"RELEASED", "CLEAR CONTRACT"} else 15
        draw.text((x, 92), label, fill="#C0C8D4", font=_font(header_size, True))

    old_rules = {
        "pusht": "Pusher + block",
        "cube": "Position only",
        "reacher": "Linear angles",
        "tworoom": "Endpoint only",
    }
    clear_rules = {
        "pusht": "Object pose",
        "cube": "24-fold pose",
        "reacher": "Wrapped joints",
        "tworoom": "Legal route",
    }
    for index, task in enumerate(TASKS):
        y = 128 + index * 102
        draw.rectangle((700, y, 1560, y + 82), fill="#111B2D")
        draw.rectangle((700, y, 708, y + 82), fill=TASK_COLORS[task])
        draw.text(
            (728, y + 24),
            TASK_LABELS[task],
            fill="#FFFFFF",
            font=_font(22, True),
        )
        draw.text(
            (865, y + 26),
            old_rules[task],
            fill="#C2CAD6",
            font=_font(20, True),
        )
        draw.text((1048, y + 23), "→", fill="#FFFFFF", font=_font(27, True))
        draw.text(
            (1090, y + 25),
            clear_rules[task],
            fill=TASK_COLORS[task],
            font=_font(21, True),
        )
        model = _success_rate(task, "strict", "official-lewm")
        random = _success_rate(task, "strict", "random")
        draw.text(
            (1325, y + 15),
            f"{model:.0f} / {random:.0f}",
            fill="#FFFFFF",
            font=_font(24, True),
        )
        draw.text(
            (1327, y + 51),
            "model / random",
            fill="#AEB7C5",
            font=_font(12, True),
        )
        draw.text(
            (1478, y + 21),
            f"{_fast_speedup(task):.2f}x",
            fill="#80E1D3",
            font=_font(20, True),
        )

    output = ASSETS / "readme_hero_v03_fast.png"
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
    print(ASSETS / "readme_hero_v03_fast.png")
    print(ASSETS / "headline_results.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
