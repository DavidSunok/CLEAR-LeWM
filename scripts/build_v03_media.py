#!/usr/bin/env python3
"""Build the v0.3 four-task README video and per-task HD GIFs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
TRACE_SOURCE = ASSETS / "media_sources" / "task_trace_summaries.json"
TASKS = ("pusht", "cube", "reacher", "tworoom")
LABELS = {
    "pusht": "PushT",
    "cube": "Cube",
    "reacher": "Reacher",
    "tworoom": "TwoRoom",
}
COLORS = {
    "pusht": "#25B6A2",
    "cube": "#F26B5E",
    "reacher": "#E5B94F",
    "tworoom": "#65AE6E",
}
ISSUES = {
    "pusht": "Pusher endpoint contaminates object success",
    "cube": "Raw quaternion rejects equivalent cube poses",
    "reacher": "Arrival and stable holding are different claims",
    "tworoom": "Endpoint success can credit wall penetration",
}
FIXES = {
    "pusht": "Object pose only + block-translation difficulty",
    "cube": "Position + 24-fold rotational symmetry",
    "reacher": "Wrapped first-hit SR + separate hold diagnostic",
    "tworoom": "Swept disk + full-radius door + route gate",
}
DATASETS = {
    "pusht": "pusht_expert_train.h5",
    "cube": "datasets/ogbench/cube_single_expert.h5",
    "reacher": "datasets/reacher.h5",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)


F = {
    "hero": font(56, True),
    "title": font(34, True),
    "h2": font(25, True),
    "body": font(20),
    "body_bold": font(20, True),
    "small": font(15),
    "small_bold": font(15, True),
    "metric": font(42, True),
}


def load_episode_frames(
    path: Path, episode_index: int, count: int
) -> list[Image.Image]:
    with h5py.File(path, "r") as dataset:
        steps = np.asarray(dataset["step_idx"])
        starts = np.flatnonzero(steps == 0)
        index = min(episode_index, len(starts) - 1)
        start = int(starts[index])
        end = int(starts[index + 1]) if index + 1 < len(starts) else len(steps)
        offsets = np.linspace(0, max(end - start - 1, 0), count).astype(int)
        pixels = np.asarray(dataset["pixels"][start + offsets])
    return [Image.fromarray(frame).convert("RGB") for frame in pixels]


def rounded(draw, box, fill, outline=None, width=1, radius=12):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_text(draw, position, text, max_width, fill, base_size=20, bold=False):
    size = base_size
    text_font = font(size, bold)
    while size > 11 and draw.textlength(text, font=text_font) > max_width:
        size -= 1
        text_font = font(size, bold)
    draw.text(position, text, font=text_font, fill=fill)


def line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[tuple[str, str, np.ndarray]],
    progress: float,
    y_max: float,
    boundaries: list[tuple[float, str, str]],
    y_label: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill="#FFFFFF")
    for fraction in np.linspace(0, 1, 5):
        y = y1 - fraction * (y1 - y0)
        draw.line((x0, y, x1, y), fill="#E5E9EF", width=1)
        draw.text(
            (x0 - 42, y - 8),
            f"{fraction * y_max:.2g}",
            font=F["small"],
            fill="#69727D",
        )
    for boundary_index, (value, color, label) in enumerate(boundaries):
        y = y1 - np.clip(value / y_max, 0, 1) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=color, width=2)
        label_y = y - 19
        if len(boundaries) > 1:
            label_y = y - 28 if boundary_index == 0 else y + 2
        draw.text((x1 - 120, label_y), label, font=F["small_bold"], fill=color)
    length = max(len(values) for _, _, values in series)
    shown = max(2, int(round(progress * (length - 1))) + 1)
    for label, color, values in series:
        values = values[:shown]
        points = []
        for index, value in enumerate(values):
            x = x0 + index / max(length - 1, 1) * (x1 - x0)
            y = y1 - np.clip(value / y_max, 0, 1) * (y1 - y0)
            points.append((x, y))
        draw.line(points, fill=color, width=4, joint="curve")
    draw.text((x0, y1 + 14), "Environment step", font=F["small"], fill="#69727D")
    draw.text((x0, y0 - 31), y_label, font=F["small_bold"], fill="#344054")
    cursor = x0
    for label, color, _ in series:
        draw.rectangle((cursor, y0 - 60, cursor + 18, y0 - 42), fill=color)
        draw.text((cursor + 27, y0 - 62), label, font=F["small"], fill="#344054")
        cursor += int(draw.textlength(label, font=F["small"])) + 66


def metric_badge(draw, x: int, y: int, label: str, values: list[float], color: str):
    rounded(draw, (x, y, x + 196, y + 58), "#FFFFFF", "#D9DEE6", 2, 10)
    draw.text((x + 13, y + 8), label, font=F["small_bold"], fill="#667085")
    draw.text(
        (x + 13, y + 28),
        f"{values[0]:g}% / {values[1]:g}%",
        font=F["body_bold"],
        fill=color,
    )


def task_frame(
    task: str,
    rgb: Image.Image,
    trace: dict,
    results: dict,
    progress: float,
) -> Image.Image:
    canvas = Image.new("RGB", (960, 540), "#F4F6F9")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 960, 76), fill="#101828")
    draw.rectangle((0, 0, 12, 76), fill=COLORS[task])
    draw.text((28, 14), LABELS[task], font=F["title"], fill="#FFFFFF")
    fit_text(draw, (220, 23), ISSUES[task], 700, "#CBD2DD", 19)

    rounded(draw, (24, 98, 374, 496), "#FFFFFF", "#D9DEE6", 2, 10)
    image = ImageOps.fit(rgb, (318, 318), method=Image.Resampling.LANCZOS)
    canvas.paste(image, (40, 116))
    draw.text(
        (40, 450), "Canonical RGB task footage", font=F["small_bold"], fill="#344054"
    )
    draw.text(
        (40, 472),
        "Visual context, not a substituted trace",
        font=F["small"],
        fill="#69727D",
    )

    steps = trace["steps"]
    if task == "pusht":
        corrected = np.asarray(
            [
                max(s["block_position_error"] / 20, s["block_angle_error_deg"] / 20)
                for s in steps
            ]
        )
        released = np.asarray(
            [
                max(s["combined_position_error"] / 20, s["block_angle_error_deg"] / 20)
                for s in steps
            ]
        )
        line_chart(
            draw,
            (438, 242, 918, 410),
            [
                ("Released: pusher + block", "#8C939D", released),
                ("CLEAR: block pose", COLORS[task], corrected),
            ],
            progress,
            max(2.5, float(np.percentile(released, 95))),
            [(1.0, "#D94747", "success")],
            "Normalized error (success < 1)",
        )
    elif task == "cube":
        raw = np.asarray(
            [
                max(s["position_error_m"] / 0.04, s["raw_orientation_error_deg"] / 30)
                for s in steps
            ]
        )
        symmetric = np.asarray(
            [
                max(
                    s["position_error_m"] / 0.04,
                    s["symmetry_orientation_error_deg"] / 30,
                )
                for s in steps
            ]
        )
        line_chart(
            draw,
            (438, 242, 918, 410),
            [
                ("Raw quaternion", "#E17B35", raw),
                ("24-fold symmetry", COLORS[task], symmetric),
            ],
            progress,
            max(3.5, float(np.percentile(raw, 95))),
            [(1.0, "#D94747", "success")],
            "Normalized pose error (success <= 1)",
        )
    else:
        error = np.asarray([s["max_wrapped_joint_error_rad"] for s in steps])
        line_chart(
            draw,
            (438, 242, 918, 410),
            [("Wrapped max joint error", "#1976B9", error)],
            progress,
            max(0.25, float(np.percentile(error, 95))),
            [(0.075, "#D9902F", "moderate"), (0.05, "#D94747", "strict")],
            "Maximum joint error (rad)",
        )

    draw.text((420, 101), "CLEAR v0.3 correction", font=F["h2"], fill="#101828")
    fit_text(draw, (420, 137), FIXES[task], 515, COLORS[task], 19, True)
    metric_badge(
        draw, 420, 454, "MODERATE  model / random", results["moderate"], COLORS[task]
    )
    metric_badge(
        draw, 628, 454, "STRICT  model / random", results["strict"], COLORS[task]
    )
    return canvas


def save_gif(frames: list[Image.Image], output: Path, duration: int = 85):
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )


def intro_frame() -> Image.Image:
    image = Image.new("RGB", (1920, 1080), "#0B1220")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, 1080), fill="#25B6A2")
    draw.text((92, 116), "CLEAR-LeWM v0.3", font=font(76, True), fill="#FFFFFF")
    draw.text(
        (96, 228),
        "Evaluation should measure the task, not an implementation accident.",
        font=font(34),
        fill="#CDD5E0",
    )
    items = (
        ("OBJECT", "Task-semantic success", "#25B6A2"),
        ("SYMMETRY", "Equivalent states stay equivalent", "#F26B5E"),
        ("DYNAMICS", "Arrival and stability are explicit", "#E5B94F"),
        ("TOPOLOGY", "No credit through walls", "#65AE6E"),
    )
    for index, (title, body, color) in enumerate(items):
        x = 96 + index * 448
        rounded(draw, (x, 440, x + 400, 700), "#FFFFFF", radius=8)
        draw.rectangle((x, 440, x + 400, 451), fill=color)
        draw.text((x + 28, 486), title, font=F["h2"], fill="#101828")
        fit_text(draw, (x + 28, 550), body, 340, "#56606D", 24)
    draw.text(
        (96, 900),
        "Official compatibility remains intact. "
        "Moderate and Strict state stronger claims.",
        font=font(28, True),
        fill="#82E1D4",
    )
    return image


def overview_task_frame(task: str, frame: Image.Image, results: dict) -> Image.Image:
    image = Image.new("RGB", (1920, 1080), "#EEF1F5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1920, 114), fill="#0B1220")
    draw.rectangle((0, 0, 16, 114), fill=COLORS[task])
    draw.text((60, 24), f"{LABELS[task]} audit", font=font(46, True), fill="#FFFFFF")
    draw.text((420, 42), ISSUES[task], font=F["h2"], fill="#CCD4DF")
    large = ImageOps.fit(
        frame.convert("RGB"), (1440, 810), method=Image.Resampling.LANCZOS
    )
    image.paste(large, (48, 160))
    rounded(draw, (1520, 160, 1870, 970), "#FFFFFF", "#D7DDE6", 2, 8)
    draw.text((1550, 198), "WHAT CHANGED", font=F["small_bold"], fill="#667085")
    fit_text(draw, (1550, 240), FIXES[task], 285, COLORS[task], 27, True)
    draw.line((1550, 390, 1840, 390), fill="#E2E6EC", width=2)
    for index, tier in enumerate(("moderate", "strict")):
        y = 440 + index * 190
        draw.text((1550, y), tier.upper(), font=F["small_bold"], fill="#667085")
        values = results[tier]
        draw.text(
            (1550, y + 35), f"{values[0]:g}%", font=F["metric"], fill=COLORS[task]
        )
        draw.text((1690, y + 51), "model", font=F["small"], fill="#667085")
        draw.text(
            (1550, y + 103),
            f"{values[1]:g}% random",
            font=F["body_bold"],
            fill="#344054",
        )
    draw.text((1550, 876), "100 fixed pairs", font=F["small_bold"], fill="#344054")
    draw.text(
        (1550, 904), "seed 42 | 300 x 30 | batch 1", font=F["small"], fill="#667085"
    )
    return image


def summary_frame(results: dict) -> Image.Image:
    image = Image.new("RGB", (1920, 1080), "#F4F6F9")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1920, 128), fill="#0B1220")
    draw.text(
        (70, 34), "A benchmark with an audit trail", font=font(44, True), fill="#FFFFFF"
    )
    draw.text((70, 174), "STRICT MODEL / RANDOM", font=F["h2"], fill="#667085")
    for index, task in enumerate(TASKS):
        x = 70 + index * 458
        rounded(draw, (x, 235, x + 410, 670), "#FFFFFF", "#D9DEE6", 2, 8)
        draw.rectangle((x, 235, x + 410, 246), fill=COLORS[task])
        draw.text((x + 26, 282), LABELS[task], font=F["title"], fill="#101828")
        values = (
            results[task]["strict"]
            if "strict" in results[task]
            else results[task]["strict_swept"]
        )
        draw.text(
            (x + 26, 375), f"{values[0]:g}%", font=font(58, True), fill=COLORS[task]
        )
        draw.text((x + 200, 402), "model", font=F["body"], fill="#667085")
        draw.text((x + 26, 500), f"{values[1]:g}% random", font=F["h2"], fill="#344054")
        draw.text((x + 26, 590), FIXES[task], font=F["small_bold"], fill="#667085")
    draw.text(
        (70, 806),
        "Pair-locked  /  checkpoint-hashed  /  runtime-fingerprinted  /  "
        "batch-1 reference",
        font=font(27, True),
        fill="#101828",
    )
    draw.text(
        (70, 874),
        "Click the README video for the full 1080p comparison and task-by-task guides.",
        font=font(24),
        fill="#667085",
    )
    return image


def build_overview(results: dict, output: Path, preview: Path, poster: Path):
    gifs = {task: Image.open(ASSETS / "task_gifs" / f"{task}.gif") for task in TASKS}
    fps = 30
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x1080",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    preview_frames: list[Image.Image] = []
    frame_number = 0

    def emit(frame: Image.Image):
        nonlocal frame_number
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        if frame_number % 4 == 0:
            preview_frames.append(frame.resize((960, 540), Image.Resampling.LANCZOS))
        frame_number += 1

    for _ in range(2 * fps):
        emit(intro_frame())
    for task in TASKS:
        gif = gifs[task]
        for index in range(4 * fps):
            gif.seek(int(index / (4 * fps) * gif.n_frames) % gif.n_frames)
            emit(overview_task_frame(task, gif.convert("RGB"), results[task]))
    final = summary_frame(results)
    for _ in range(3 * fps):
        emit(final)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while building overview")
    final.save(poster, optimize=True)
    save_gif(preview_frames, preview, duration=133)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(TRACE_SOURCE.read_text())
    results = source["results"]
    episode_indices = {"pusht": 4, "cube": 5, "reacher": 6}
    for task in ("pusht", "cube", "reacher"):
        rgb_frames = load_episode_frames(
            args.cache_dir / DATASETS[task], episode_indices[task], 60
        )
        frames = [
            task_frame(
                task,
                rgb_frames[index],
                source[task],
                results[task],
                index / 59,
            )
            for index in range(60)
        ]
        save_gif(frames, ASSETS / "task_gifs" / f"{task}.gif")
    overview_results = {
        **{task: results[task] for task in ("pusht", "cube", "reacher")},
        "tworoom": {
            "moderate": results["tworoom"].get("moderate", [0, 0]),
            "strict": results["tworoom"]["strict_swept"],
            "strict_swept": results["tworoom"]["strict_swept"],
        },
    }
    showcase = ASSETS / "showcase"
    showcase.mkdir(parents=True, exist_ok=True)
    build_overview(
        overview_results,
        showcase / "clear_lewm_v03_overview_1080p.mp4",
        showcase / "clear_lewm_v03_overview_preview.gif",
        showcase / "clear_lewm_v03_overview_poster.png",
    )
    for path in sorted((ASSETS / "task_gifs").glob("*.gif")):
        print(path)
    print(showcase / "clear_lewm_v03_overview_1080p.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
