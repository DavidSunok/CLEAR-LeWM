#!/usr/bin/env python3
"""Build the v0.3 README comparison film and four high-resolution task GIFs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clear_lewm.tasks import (  # noqa: E402
    cube_symmetry_angle_deg,
    quaternion_angle_deg,
    wrapped_angle_error,
)
from clear_lewm.topology import TwoRoomGeometry  # noqa: E402

ASSETS = ROOT / "assets"
TRACE_SOURCE = ASSETS / "media_sources" / "task_trace_summaries.json"
TWOROOM_COMPARISON = ASSETS / "media_sources" / "tworoom_rule_comparison.json"
TWOROOM_RESULT = (
    ROOT / "results" / "v0.3" / "tworoom-strict-official-lewm-seed42-n100.json"
)
TASKS = ("pusht", "cube", "reacher", "tworoom")
LABELS = {
    "pusht": "PushT",
    "cube": "Cube",
    "reacher": "Reacher",
    "tworoom": "TwoRoom",
}
DOMAINS = {
    "pusht": "OBJECT-LEVEL SUCCESS",
    "cube": "SYMMETRY-AWARE POSE",
    "reacher": "PERIODIC JOINT STATE",
    "tworoom": "TOPOLOGY-VALID ROUTE",
}
COLORS = {
    "pusht": "#25B6A2",
    "cube": "#F26B5E",
    "reacher": "#E5B94F",
    "tworoom": "#65AE6E",
}
ISSUES = {
    "pusht": "Pusher position can overturn a correct object placement.",
    "cube": "Position alone ignores pose; raw rotation ignores cube symmetry.",
    "reacher": "Linear angle distance breaks at the periodic boundary.",
    "tworoom": "Endpoint proximity can hide an impossible route through a wall.",
}
FIXES = {
    "pusht": "Measure the T block pose and block displacement only.",
    "cube": "Match position and minimize rotation over 24 cube symmetries.",
    "reacher": "Use wrapped joint error; report holding as a separate diagnostic.",
    "tworoom": "Sweep the full agent disk and require a legal door crossing.",
}
SETTINGS = {
    "pusht": "100 fixed pairs  |  300 x 30 CEM  |  batch 1",
    "cube": "100 fixed pairs  |  300 x 30 CEM  |  batch 1",
    "reacher": "100 fixed pairs  |  300 x 30 CEM  |  batch 1",
    "tworoom": "100 cross-room pairs  |  swept disk  |  batch 1",
}
DATASETS = {
    "pusht": "pusht_expert_train.h5",
    "cube": "datasets/ogbench/cube_single_expert.h5",
    "reacher": "datasets/reacher.h5",
}

TASK_WIDTH, TASK_HEIGHT = 1200, 675
VIDEO_WIDTH, VIDEO_HEIGHT = 1920, 1080
BG = "#F4F6F9"
INK = "#101828"
MUTED = "#667085"
GRID = "#E2E7ED"
RED = "#D94747"
RED_BG = "#FBE9E7"
GREEN = "#12866F"
GREEN_BG = "#E4F5F0"
PURPLE = "#B45CFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)


F = {
    "task": font(38, True),
    "title": font(28, True),
    "h2": font(23, True),
    "body": font(18),
    "body_bold": font(18, True),
    "small": font(14),
    "small_bold": font(14, True),
    "metric": font(28, True),
}


def rounded(draw, box, fill, outline=None, width=1, radius=10) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw, center, text, text_font, fill=INK) -> None:
    bounds = draw.textbbox((0, 0), text, font=text_font)
    draw.text(
        (
            center[0] - (bounds[2] - bounds[0]) / 2,
            center[1] - (bounds[3] - bounds[1]) / 2,
        ),
        text,
        font=text_font,
        fill=fill,
    )


def fit_text(
    draw,
    position,
    text,
    max_width,
    fill,
    base_size=18,
    bold=False,
    minimum=12,
) -> None:
    size = base_size
    text_font = font(size, bold)
    while size > minimum and draw.textlength(text, font=text_font) > max_width:
        size -= 1
        text_font = font(size, bold)
    draw.text(position, text, font=text_font, fill=fill)


def wrap_lines(draw, text: str, text_font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=text_font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw,
    position,
    text,
    max_width,
    text_font,
    fill,
    line_gap=6,
    max_lines=3,
) -> None:
    lines = wrap_lines(draw, text, text_font, max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while draw.textlength(lines[-1] + "...", font=text_font) > max_width:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] += "..."
    line_height = draw.textbbox((0, 0), "Ag", font=text_font)[3]
    for index, line in enumerate(lines):
        draw.text(
            (position[0], position[1] + index * (line_height + line_gap)),
            line,
            font=text_font,
            fill=fill,
        )


def load_episode_data(
    path: Path,
    episode_index: int,
    count: int,
    keys: tuple[str, ...],
) -> dict[str, np.ndarray | list[Image.Image]]:
    with h5py.File(path, "r") as dataset:
        if "ep_offset" in dataset and "ep_len" in dataset:
            index = min(episode_index, len(dataset["ep_offset"]) - 1)
            start = int(dataset["ep_offset"][index])
            end = start + int(dataset["ep_len"][index])
        else:
            steps = np.asarray(dataset["step_idx"])
            starts = np.flatnonzero(steps == 0)
            index = min(episode_index, len(starts) - 1)
            start = int(starts[index])
            end = int(starts[index + 1]) if index + 1 < len(starts) else len(steps)
        offsets = np.linspace(0, max(end - start - 1, 0), count).astype(int)
        payload = {key: np.asarray(dataset[key][start:end])[offsets] for key in keys}
        pixels = np.asarray(dataset["pixels"][start:end])[offsets]
    payload["frames"] = [Image.fromarray(frame).convert("RGB") for frame in pixels]
    return payload


def synchronized_metric_trace(
    task: str, payload: dict[str, np.ndarray | list[Image.Image]]
) -> dict:
    if task == "pusht":
        states = np.asarray(payload["state"], dtype=np.float64)
        goal = states[-1]
        agent = np.linalg.norm(states[:, :2] - goal[None, :2], axis=1)
        combined = np.linalg.norm(states[:, :4] - goal[None, :4], axis=1)
        block = np.linalg.norm(states[:, 2:4] - goal[None, 2:4], axis=1)
        angle = np.degrees(
            np.abs(
                np.arctan2(
                    np.sin(states[:, 4] - goal[4]),
                    np.cos(states[:, 4] - goal[4]),
                )
            )
        )
        steps = [
            {
                "combined_position_error": float(combined[index]),
                "agent_position_error": float(agent[index]),
                "block_position_error": float(block[index]),
                "block_angle_error_deg": float(angle[index]),
            }
            for index in range(len(states))
        ]
        corrected = np.maximum(block / 20.0, angle / 20.0)
        released = np.maximum(combined / 20.0, angle / 20.0)
        contrast = np.flatnonzero((corrected < 1.0) & (released >= 1.0))
    elif task == "cube":
        position = np.asarray(payload["privileged_block_0_pos"], dtype=np.float64)
        target_position = np.asarray(
            payload["privileged_target_block_pos"], dtype=np.float64
        )
        quaternion = np.asarray(payload["privileged_block_0_quat"], dtype=np.float64)
        yaw = np.asarray(
            payload["privileged_target_block_yaw"], dtype=np.float64
        ).reshape(-1)
        zeros = np.zeros_like(yaw)
        target_quaternion = np.stack(
            (np.cos(yaw / 2), zeros, zeros, np.sin(yaw / 2)), axis=-1
        )
        position_error = np.linalg.norm(position - target_position, axis=1)
        raw_error = quaternion_angle_deg(quaternion, target_quaternion)
        symmetry_error = cube_symmetry_angle_deg(quaternion, target_quaternion)
        steps = [
            {
                "position_error_m": float(position_error[index]),
                "raw_orientation_error_deg": float(raw_error[index]),
                "symmetry_orientation_error_deg": float(symmetry_error[index]),
            }
            for index in range(len(position))
        ]
        contrast = np.flatnonzero((position_error <= 0.04) & (symmetry_error > 30.0))
    elif task == "reacher":
        qpos = np.asarray(payload["qpos"], dtype=np.float64)
        goal = qpos[-1]
        raw_per_joint = np.abs(qpos - goal[None])
        periodic_per_joint = wrapped_angle_error(qpos, goal[None])
        raw_error = np.max(raw_per_joint, axis=1)
        periodic_error = np.max(periodic_per_joint, axis=1)
        steps = [
            {
                "max_raw_joint_error_rad": float(raw_error[index]),
                "max_wrapped_joint_error_rad": float(periodic_error[index]),
                "joint_1_raw_error_rad": float(raw_per_joint[index, 0]),
                "joint_1_wrapped_error_rad": float(periodic_per_joint[index, 0]),
                "joint_1_current_angle_deg": float(np.degrees(qpos[index, 0])),
                "joint_1_target_angle_deg": float(np.degrees(goal[0])),
            }
            for index in range(len(qpos))
        ]
        contrast = np.asarray([], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported synchronized task: {task}")
    trace = {"steps": steps}
    if len(contrast):
        trace["comparison_end_index"] = int(contrast[-1])
    return trace


def line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[tuple[str, str, np.ndarray]],
    progress: float,
    y_max: float,
    boundary: tuple[float, str, str],
    y_label: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill="#FFFFFF")
    for fraction in np.linspace(0, 1, 5):
        y = y1 - fraction * (y1 - y0)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        value = fraction * y_max
        label = f"{value:.2f}" if y_max < 1 else f"{value:.1f}"
        draw.text((x0 - 47, y - 8), label, font=F["small"], fill=MUTED)
    value, color, label = boundary
    y = y1 - np.clip(value / y_max, 0, 1) * (y1 - y0)
    draw.line((x0, y, x1, y), fill=color, width=2)
    label_width = draw.textlength(label, font=F["small_bold"])
    draw.rectangle((x1 - label_width - 14, y - 23, x1 + 4, y - 3), fill="#FFFFFF")
    draw.text(
        (x1 - label_width - 5, y - 23),
        label,
        font=F["small_bold"],
        fill=color,
    )
    length = max(len(values) for _, _, values in series)
    location = np.clip(progress, 0.0, 1.0) * (length - 1)
    whole = int(np.floor(location))
    fraction = location - whole
    for _label, line_color, values in series:
        values = np.asarray(values, dtype=np.float64)
        visible_indices = list(range(whole + 1))
        visible_values = list(values[: whole + 1])
        if whole < length - 1:
            interpolated = values[whole] * (1 - fraction) + values[whole + 1] * fraction
            visible_indices.append(location)
            visible_values.append(float(interpolated))
        points = []
        for index, current in zip(visible_indices, visible_values, strict=True):
            x = x0 + index / max(length - 1, 1) * (x1 - x0)
            y = y1 - np.clip(current / y_max, 0, 1) * (y1 - y0)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=line_color, width=4, joint="curve")
        if points:
            x, y = points[-1]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=line_color)
    draw.text((x0, y1 + 12), "Environment step", font=F["small"], fill=MUTED)
    draw.text((x0, y0 - 31), y_label, font=F["small_bold"], fill="#344054")
    cursor = x0
    for label, line_color, _ in series:
        draw.line((cursor, y0 - 57, cursor + 22, y0 - 57), fill=line_color, width=5)
        draw.text((cursor + 31, y0 - 68), label, font=F["small"], fill="#344054")
        cursor += int(draw.textlength(label, font=F["small"])) + 78


def result_card(draw, box, title: str, value: str, color: str) -> None:
    rounded(draw, box, "#FFFFFF", "#D4DAE3", 2, 8)
    x0, y0, _, _ = box
    draw.text((x0 + 16, y0 + 10), title, font=F["small_bold"], fill=MUTED)
    fit_text(draw, (x0 + 16, y0 + 35), value, box[2] - box[0] - 32, color, 22, True)


def setting_card(draw, box, task: str) -> None:
    rounded(draw, box, "#101828", radius=8)
    x0, y0, _, _ = box
    draw.text(
        (x0 + 16, y0 + 10), "REFERENCE SETTING", font=F["small_bold"], fill="#98A2B3"
    )
    fit_text(
        draw,
        (x0 + 16, y0 + 37),
        SETTINGS[task],
        box[2] - box[0] - 32,
        "#FFFFFF",
        16,
        True,
    )


def _geometry(record: dict) -> TwoRoomGeometry:
    values = dict(record["geometry"])
    values["doors"] = tuple(tuple(item) for item in values["doors"])
    return TwoRoomGeometry(**values)


def _map_point(point, box, image_size: float) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (
        x0 + float(point[0]) / image_size * (x1 - x0),
        y0 + float(point[1]) / image_size * (y1 - y0),
    )


def _path_prefix(record: dict, progress: float) -> np.ndarray:
    positions = np.asarray(record["positions"], dtype=np.float64)
    if len(positions) < 2:
        return positions.copy()
    location = np.clip(progress, 0.0, 1.0) * (len(positions) - 1)
    index = min(int(location), len(positions) - 2)
    fraction = location - index
    point = positions[index] * (1 - fraction) + positions[index + 1] * fraction
    return np.concatenate((positions[: index + 1], point[None]), axis=0)


def draw_tworoom_world(
    draw,
    box,
    record: dict,
    progress: float,
    path_color: str,
    invalid_color: str | None = None,
) -> None:
    geom = _geometry(record)
    x0, y0, x1, y1 = box
    rounded(draw, (x0 - 2, y0 - 2, x1 + 2, y1 + 2), "#FFFFFF", "#D4DAE3", 2, 8)
    for rectangle in geom.solid_wall_rectangles():
        low = _map_point((rectangle.xmin, rectangle.ymin), box, geom.image_size)
        high = _map_point((rectangle.xmax, rectangle.ymax), box, geom.image_size)
        draw.rectangle((*low, *high), fill="#242A30")
    low, high = geom.center_bounds
    a = _map_point((low, low), box, geom.image_size)
    b = _map_point((high, high), box, geom.image_size)
    draw.rectangle((*a, *b), outline="#8A949E", width=2)
    wall_low, wall_high = geom.wall_bounds
    for door_low, door_high in geom.clear_door_intervals():
        if geom.normal_axis == 0:
            a = _map_point((wall_low, door_low), box, geom.image_size)
            b = _map_point((wall_high, door_high), box, geom.image_size)
        else:
            a = _map_point((door_low, wall_low), box, geom.image_size)
            b = _map_point((door_high, wall_high), box, geom.image_size)
        draw.rectangle((*a, *b), fill="#CDEFE6", outline=GREEN, width=2)
    goal_x, goal_y = _map_point(record["goal_position"], box, geom.image_size)
    draw.ellipse(
        (goal_x - 11, goal_y - 11, goal_x + 11, goal_y + 11),
        fill="#FFFFFF",
        outline=GREEN,
        width=4,
    )
    draw.line((goal_x - 6, goal_y, goal_x + 6, goal_y), fill=GREEN, width=3)
    draw.line((goal_x, goal_y - 6, goal_x, goal_y + 6), fill=GREEN, width=3)
    points = _path_prefix(record, progress)
    mapped = [_map_point(point, box, geom.image_size) for point in points]
    if len(mapped) > 1:
        draw.line(mapped, fill=path_color, width=5, joint="curve")
    if mapped:
        x, y = mapped[-1]
        scale = (x1 - x0) / geom.image_size
        radius = geom.agent_radius * scale
        outline = invalid_color or path_color
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="#FFFFFF",
            outline=outline,
            width=4,
        )
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=outline)


def task_frame(
    task: str,
    rgb: Image.Image | None,
    trace: dict,
    results: dict,
    progress: float,
) -> Image.Image:
    canvas = Image.new("RGB", (TASK_WIDTH, TASK_HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, TASK_WIDTH, 88), fill="#101828")
    draw.rectangle((0, 0, 12, 88), fill=COLORS[task])
    draw.text((32, 17), LABELS[task], font=F["task"], fill="#FFFFFF")
    domain_x = 32 + draw.textlength(LABELS[task], font=F["task"]) + 38
    draw.text((domain_x, 31), DOMAINS[task], font=F["small_bold"], fill=COLORS[task])
    fit_text(draw, (635, 31), ISSUES[task], 530, "#CDD5DF", 16)

    rounded(draw, (30, 112, 398, 552), "#FFFFFF", "#D4DAE3", 2, 9)
    visual_box = (50, 132, 378, 460)
    if task == "tworoom":
        draw_tworoom_world(draw, visual_box, trace, progress, COLORS[task])
        draw.text(
            (50, 479), "Legal cross-room rollout", font=F["small_bold"], fill="#344054"
        )
        draw.text(
            (50, 505), "Full agent disk shown to scale", font=F["small"], fill=MUTED
        )
    else:
        assert rgb is not None
        image = ImageOps.fit(rgb, (328, 328), method=Image.Resampling.LANCZOS)
        canvas.paste(image, (50, 132))
        draw.text(
            (50, 479),
            "Canonical task observation",
            font=F["small_bold"],
            fill="#344054",
        )
        draw.text(
            (50, 505),
            "RGB context with a verified metric trace",
            font=F["small"],
            fill=MUTED,
        )

    draw.text((432, 116), "What CLEAR measures", font=F["title"], fill=INK)
    draw_wrapped(
        draw, (432, 157), FIXES[task], 725, F["body_bold"], COLORS[task], max_lines=2
    )

    steps = trace["steps"] if task != "tworoom" else None
    if task == "pusht":
        released = np.asarray(
            [
                max(s["combined_position_error"] / 20, s["block_angle_error_deg"] / 20)
                for s in steps
            ]
        )
        corrected = np.asarray(
            [
                max(s["block_position_error"] / 20, s["block_angle_error_deg"] / 20)
                for s in steps
            ]
        )
        line_chart(
            draw,
            (490, 324, 1160, 502),
            [
                ("Pusher + block", "#9199A4", released),
                ("Block pose", COLORS[task], corrected),
            ],
            progress,
            max(2.8, float(np.percentile(released, 95))),
            (1.0, RED, "success < 1"),
            "Normalized error",
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
            (490, 324, 1160, 502),
            [
                ("Raw rotation", "#7D8794", raw),
                ("24-fold pose", COLORS[task], symmetric),
            ],
            progress,
            max(3.6, float(np.percentile(raw, 95))),
            (1.0, RED, "success <= 1"),
            "Normalized pose error",
        )
    elif task == "reacher":
        error = np.asarray([s["max_wrapped_joint_error_rad"] for s in steps])
        line_chart(
            draw,
            (490, 324, 1160, 502),
            [("Wrapped joint error", "#257DB2", error)],
            progress,
            max(0.24, float(np.percentile(error, 95))),
            (0.075, RED, "success < 0.075"),
            "Maximum wrapped error (rad)",
        )
    else:
        positions = np.asarray(trace["positions"], dtype=np.float64)
        goal = np.asarray(trace["goal_position"], dtype=np.float64)
        distance = np.linalg.norm(positions - goal[None], axis=1)
        line_chart(
            draw,
            (490, 324, 1160, 502),
            [("Goal distance", COLORS[task], distance)],
            progress,
            max(45.0, float(np.percentile(distance, 95))),
            (8.0, RED, "strict < 8 px"),
            "Endpoint distance (px), route valid",
        )

    result_card(
        draw,
        (30, 575, 360, 650),
        "MODERATE  model / random",
        f"{results['moderate'][0]:g}% / {results['moderate'][1]:g}%",
        COLORS[task],
    )
    result_card(
        draw,
        (378, 575, 708, 650),
        "STRICT  model / random",
        f"{results['strict'][0]:g}% / {results['strict'][1]:g}%",
        COLORS[task],
    )
    setting_card(draw, (726, 575, 1170, 650), task)
    return canvas


def _global_palette(frames: list[Image.Image], colors: int) -> Image.Image:
    thumb_w, thumb_h = 200, 113
    columns = 6
    rows = int(np.ceil(len(frames) / columns))
    atlas = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "white")
    for index, frame in enumerate(frames):
        atlas.paste(
            frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS),
            ((index % columns) * thumb_w, (index // columns) * thumb_h),
        )
    return atlas.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)


def save_gif(
    frames: list[Image.Image], output: Path, duration: int = 85, colors: int = 192
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    palette = _global_palette(frames, colors)
    encoded = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]
    encoded[0].save(
        output,
        save_all=True,
        append_images=encoded[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )


def intro_frame() -> Image.Image:
    image = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "#0B1220")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, VIDEO_HEIGHT), fill=PURPLE)
    draw.text((92, 92), "CLEAR-LeWM v0.3", font=font(72, True), fill="#FFFFFF")
    draw.text(
        (96, 206), "WHEN SUCCESS MEANS COMPLETION", font=font(39, True), fill=PURPLE
    )
    draw_wrapped(
        draw,
        (96, 282),
        "Four released or naive judgements on the left. "
        "Four task-semantic judgements on the right.",
        1080,
        font(39, True),
        "#D0D5DD",
        line_gap=16,
        max_lines=2,
    )
    labels = (
        ("OBJECT", "PushT", COLORS["pusht"]),
        ("SYMMETRY", "Cube", COLORS["cube"]),
        ("PERIODICITY", "Reacher", COLORS["reacher"]),
        ("TOPOLOGY", "TwoRoom", COLORS["tworoom"]),
    )
    for index, (domain, task, color) in enumerate(labels):
        x = 96 + index * 448
        draw.rectangle((x, 520, x + 392, 532), fill=color)
        draw.text((x, 566), domain, font=font(26, True), fill=color)
        draw.text((x, 628), task, font=font(43, True), fill="#FFFFFF")
    draw.text(
        (96, 902),
        "Same task. Explicit rule. Reproducible judgement.",
        font=font(29, True),
        fill="#FFFFFF",
    )
    draw.text(
        (96, 960),
        "Fixed pairs  |  paired random floor  |  runtime fingerprint  |  "
        "solver batch 1",
        font=font(22),
        fill="#98A2B3",
    )
    return image


def _comparison_panel(
    draw,
    box,
    eyebrow: str,
    title: str,
    color: str,
    positive: bool,
) -> None:
    rounded(draw, box, "#FFFFFF", "#D4DAE3", 2, 10)
    x0, y0, x1, _ = box
    draw.rectangle((x0, y0, x1, y0 + 10), fill=color)
    draw.text((x0 + 30, y0 + 28), eyebrow, font=font(18, True), fill=color)
    fit_text(draw, (x0 + 30, y0 + 67), title, x1 - x0 - 60, INK, 29, True)
    badge_fill = GREEN_BG if positive else RED_BG
    badge_color = GREEN if positive else RED
    rounded(draw, (x1 - 170, y0 + 26, x1 - 30, y0 + 65), badge_fill, radius=19)
    centered(
        draw,
        (x1 - 100, y0 + 45),
        "TASK RULE" if positive else "OLD RULE",
        font(14, True),
        badge_color,
    )


def _status(draw, box, label: str, detail: str, passed: bool) -> None:
    fill = GREEN_BG if passed else RED_BG
    color = GREEN if passed else RED
    rounded(draw, box, fill, radius=8)
    x0, y0, x1, _ = box
    fit_text(
        draw,
        (x0 + 18, y0 + 11),
        label,
        x1 - x0 - 36,
        color,
        25,
        True,
        minimum=16,
    )
    fit_text(draw, (x0 + 18, y0 + 50), detail, x1 - x0 - 36, MUTED, 21, False)


def _metric_block(
    draw,
    origin,
    name: str,
    value: str,
    rule: str,
    color: str,
    max_width: int = 330,
) -> None:
    x, y = origin
    fit_text(draw, (x, y), name, max_width, MUTED, 20, True, minimum=14)
    fit_text(draw, (x, y + 44), value, max_width, color, 48, True)
    draw_wrapped(
        draw,
        (x, y + 116),
        rule,
        max_width,
        font(22),
        "#344054",
        line_gap=7,
        max_lines=3,
    )


def _interpolated_value(
    steps: list[dict], key: str, progress: float, end_index: int | None = None
) -> float:
    final = len(steps) - 1 if end_index is None else min(end_index, len(steps) - 1)
    location = np.clip(progress, 0.0, 1.0) * final
    lower = int(np.floor(location))
    upper = min(lower + 1, final)
    fraction = location - lower
    return float(steps[lower][key] * (1 - fraction) + steps[upper][key] * fraction)


def _angle_dial(
    draw,
    box,
    raw: bool,
    current_angle_deg: float,
    target_angle_deg: float,
) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    radius = min(x1 - x0, y1 - y0) * 0.34
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill="#F8FAFC",
        outline="#CBD3DE",
        width=4,
    )
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        a = (cx + np.cos(angle) * (radius - 12), cy + np.sin(angle) * (radius - 12))
        b = (cx + np.cos(angle) * radius, cy + np.sin(angle) * radius)
        draw.line((*a, *b), fill="#98A2B3", width=2)
    angles = np.deg2rad([current_angle_deg, target_angle_deg])
    dial_colors = ("#D94747", "#257DB2")
    for angle, color in zip(angles, dial_colors, strict=True):
        end = (cx + np.cos(angle) * (radius - 24), cy + np.sin(angle) * (radius - 24))
        draw.line((cx, cy, *end), fill=color, width=6)
        draw.ellipse((end[0] - 6, end[1] - 6, end[0] + 6, end[1] + 6), fill=color)
    arc_color = RED if raw else GREEN
    arc_box = (cx - radius - 22, cy - radius - 22, cx + radius + 22, cy + radius + 22)
    if raw:
        start = target_angle_deg % 360
        end = start + current_angle_deg - target_angle_deg
        while min(start, end) < 0:
            start += 360
            end += 360
        draw.arc(arc_box, min(start, end), max(start, end), fill=arc_color, width=7)
    else:
        start = current_angle_deg % 360
        delta = (target_angle_deg - current_angle_deg + 180) % 360 - 180
        end = start + delta
        while min(start, end) < 0:
            start += 360
            end += 360
        draw.arc(arc_box, min(start, end), max(start, end), fill=arc_color, width=9)
    centered(
        draw,
        (cx, y1 - 28),
        "linear subtraction" if raw else "shortest wrapped arc",
        font(17, True),
        arc_color,
    )


def comparison_frame(
    task: str,
    rgb: Image.Image | None,
    source: dict,
    comparison: dict,
    progress: float,
) -> Image.Image:
    image = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "#EEF2F6")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, VIDEO_WIDTH, 112), fill="#0B1220")
    draw.rectangle((0, 0, 16, 112), fill=COLORS[task])
    draw.text((58, 22), LABELS[task], font=font(46, True), fill="#FFFFFF")
    draw.text((330, 42), DOMAINS[task], font=font(20, True), fill=COLORS[task])
    draw.text((1170, 42), "RULE COMPARISON", font=font(18, True), fill="#98A2B3")
    draw.text((1375, 36), "HEURISTIC", font=font(24, True), fill="#F3A19A")
    draw.text((1560, 36), "→", font=font(30, True), fill="#FFFFFF")
    draw.text((1620, 36), "TASK SEMANTICS", font=font(24, True), fill="#81E2D4")

    left = (54, 148, 930, 1014)
    right = (990, 148, 1866, 1014)
    left_titles = {
        "pusht": "Pusher + block endpoint",
        "cube": "Position-only completion",
        "reacher": "Linear angle subtraction",
        "tworoom": "Endpoint collision",
    }
    right_titles = {
        "pusht": "Object pose only",
        "cube": "Position + 24-fold pose",
        "reacher": "Wrapped angular distance",
        "tworoom": "Swept disk + route gate",
    }
    _comparison_panel(draw, left, "RELEASED / NAIVE", left_titles[task], RED, False)
    _comparison_panel(draw, right, "CLEAR v0.3", right_titles[task], GREEN, True)

    if task in ("pusht", "cube"):
        assert rgb is not None
        visual = ImageOps.fit(rgb, (390, 390), method=Image.Resampling.LANCZOS)
        image.paste(visual, (88, 282))
        image.paste(visual, (1024, 282))
        steps = source[task]["steps"]
        comparison_end = source[task].get("comparison_end_index")
        display_end = len(steps) - 1 if comparison_end is None else comparison_end
        step_index = int(round(progress * display_end))
        if task == "pusht":
            angle = _interpolated_value(
                steps,
                "block_angle_error_deg",
                progress,
                comparison_end,
            )
            old_value = max(
                _interpolated_value(
                    steps,
                    "combined_position_error",
                    progress,
                    comparison_end,
                )
                / 20,
                angle / 20,
            )
            new_value = max(
                _interpolated_value(
                    steps,
                    "block_position_error",
                    progress,
                    comparison_end,
                )
                / 20,
                angle / 20,
            )
            pusher_term = (
                _interpolated_value(
                    steps,
                    "agent_position_error",
                    progress,
                    comparison_end,
                )
                / 20
            )
            old_pass, new_pass = old_value < 1, new_value < 1
            _metric_block(
                draw,
                (520, 314),
                f"NORMALIZED ERROR  ·  STEP {step_index:03d}",
                f"{old_value:.2f}",
                f"Pusher {pusher_term:.2f}; block {new_value:.2f}; both included.",
                RED,
            )
            _metric_block(
                draw,
                (1456, 314),
                f"NORMALIZED ERROR  ·  STEP {step_index:03d}",
                f"{new_value:.2f}",
                f"Block {new_value:.2f}; pusher excluded from this rule.",
                GREEN,
            )
            old_detail = "The pusher keeps changing this score after the block settles."
            new_detail = "Only block position and angle determine this output."
        else:
            position = _interpolated_value(
                steps, "position_error_m", progress, comparison_end
            )
            symmetry = _interpolated_value(
                steps,
                "symmetry_orientation_error_deg",
                progress,
                comparison_end,
            )
            old_value = position * 100
            new_value = max(position / 0.04, symmetry / 30)
            old_pass, new_pass = old_value <= 4, new_value <= 1
            _metric_block(
                draw,
                (520, 314),
                f"POSITION ERROR  ·  STEP {step_index:03d}",
                f"{old_value:.1f} cm",
                "The released rule ignores orientation.",
                RED,
            )
            _metric_block(
                draw,
                (1456, 314),
                f"NORMALIZED POSE  ·  STEP {step_index:03d}",
                f"{new_value:.2f}",
                "Position plus the nearest of 24 equivalent rotations.",
                GREEN,
            )
            old_detail = (
                "Position is close, so an unfinished pose is credited."
                if old_pass
                else "The position threshold has not been reached yet."
            )
            new_detail = "Pose must also be correct up to cube symmetry."
        _status(
            draw,
            (88, 746, 894, 846),
            f"OLD RULE OUTPUT: {'SUCCESS' if old_pass else 'NOT COMPLETE'}",
            old_detail,
            False,
        )
        _status(
            draw,
            (1024, 746, 1830, 846),
            f"CLEAR RULE OUTPUT: {'SUCCESS' if new_pass else 'NOT COMPLETE'}",
            new_detail,
            True,
        )
    elif task == "reacher":
        assert rgb is not None
        thumbnail = ImageOps.fit(rgb, (170, 170), method=Image.Resampling.LANCZOS)
        image.paste(thumbnail, (88, 282))
        image.paste(thumbnail, (1024, 282))
        steps = source[task]["steps"]
        raw_error = np.degrees(
            _interpolated_value(steps, "joint_1_raw_error_rad", progress)
        )
        wrapped_error = np.degrees(
            _interpolated_value(steps, "joint_1_wrapped_error_rad", progress)
        )
        joint_current = _interpolated_value(
            steps, "joint_1_current_angle_deg", progress
        )
        joint_target = _interpolated_value(steps, "joint_1_target_angle_deg", progress)
        _angle_dial(
            draw,
            (260, 270, 670, 700),
            raw=True,
            current_angle_deg=joint_current,
            target_angle_deg=joint_target,
        )
        _angle_dial(
            draw,
            (1196, 270, 1606, 700),
            raw=False,
            current_angle_deg=joint_current,
            target_angle_deg=joint_target,
        )
        _metric_block(
            draw,
            (680, 330),
            "JOINT 1 ERROR",
            f"{raw_error:.1f}°",
            "Raw Joint 1 error from this exact RGB frame.",
            RED,
            max_width=210,
        )
        _metric_block(
            draw,
            (1616, 330),
            "JOINT 1 WRAPPED",
            f"{wrapped_error:.1f}°",
            "Wrapped Joint 1 error from the same qpos frame.",
            GREEN,
            max_width=210,
        )
        _status(
            draw,
            (88, 746, 894, 846),
            (
                "RAW RULE OUTPUT: SUCCESS"
                if raw_error < 4.3
                else "RAW RULE OUTPUT: NOT COMPLETE"
            ),
            (
                "Raw subtraction overstates this physical joint displacement."
                if raw_error - wrapped_error > 20.0
                else "The raw error follows the actual joint trajectory."
            ),
            raw_error < 4.3,
        )
        _status(
            draw,
            (1024, 746, 1830, 846),
            (
                "WRAPPED RULE OUTPUT: SUCCESS"
                if wrapped_error < 4.3
                else "WRAPPED RULE OUTPUT: NOT COMPLETE"
            ),
            "RGB, dial, and wrapped metric use the same HDF5 qpos sample.",
            True,
        )
    else:
        old = comparison["official_endpoint"]
        new = comparison["swept_circle"]
        draw_tworoom_world(draw, (98, 276, 574, 752), old, progress, "#C87070", RED)
        draw_tworoom_world(draw, (1034, 276, 1510, 752), new, progress, "#257DB2")
        old_point = _path_prefix(old, progress)[-1]
        new_point = _path_prefix(new, progress)[-1]
        old_distance = np.linalg.norm(
            old_point - np.asarray(old["goal_position"], dtype=np.float64)
        )
        new_distance = np.linalg.norm(
            new_point - np.asarray(new["goal_position"], dtype=np.float64)
        )
        _metric_block(
            draw,
            (610, 330),
            "GOAL DISTANCE",
            f"{old_distance:.1f} px",
            "Distance falls, but route_valid remains false.",
            RED,
            max_width=250,
        )
        _metric_block(
            draw,
            (1546, 330),
            "GOAL DISTANCE",
            f"{new_distance:.1f} px",
            "Distance and route_valid are checked together.",
            GREEN,
            max_width=250,
        )
        _status(
            draw,
            (88, 790, 894, 890),
            (
                "ENDPOINT RULE OUTPUT: SUCCESS"
                if old_distance < 8.0
                else "ENDPOINT RULE OUTPUT: NOT COMPLETE"
            ),
            "This output ignores that the recorded route intersects the wall.",
            False,
        )
        _status(
            draw,
            (1024, 790, 1830, 890),
            (
                "CLEAR RULE OUTPUT: SUCCESS"
                if new_distance < 8.0
                else "CLEAR RULE OUTPUT: NOT COMPLETE"
            ),
            "Swept collision keeps the route legal through the usable opening.",
            True,
        )

    draw_wrapped(
        draw, (90, 904), ISSUES[task], 790, font(24, True), "#7A3030", max_lines=2
    )
    draw_wrapped(
        draw,
        (1026, 904),
        FIXES[task],
        790,
        font(24, True),
        "#126E60",
        max_lines=2,
    )
    return image


def summary_frame(results: dict) -> Image.Image:
    image = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "#F4F6F9")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, VIDEO_WIDTH, 132), fill="#0B1220")
    draw.text(
        (70, 30),
        "One benchmark. Four task-semantic contracts.",
        font=font(44, True),
        fill="#FFFFFF",
    )
    draw.text(
        (70, 180),
        "STRICT SR  |  LeWM MODEL / RANDOM POLICY",
        font=font(24, True),
        fill=MUTED,
    )
    for index, task in enumerate(TASKS):
        x = 70 + index * 458
        rounded(draw, (x, 238, x + 410, 700), "#FFFFFF", "#D4DAE3", 2, 8)
        draw.rectangle((x, 238, x + 410, 248), fill=COLORS[task])
        draw.text((x + 28, 286), LABELS[task], font=font(31, True), fill=INK)
        values = results[task]["strict"]
        draw.text(
            (x + 28, 380), f"{values[0]:g}%", font=font(60, True), fill=COLORS[task]
        )
        draw.text((x + 205, 412), "LeWM model", font=font(21, True), fill=MUTED)
        draw.text(
            (x + 28, 500),
            f"{values[1]:g}% random policy",
            font=font(27, True),
            fill="#344054",
        )
        draw_wrapped(
            draw,
            (x + 28, 585),
            FIXES[task],
            350,
            font(18, True),
            MUTED,
            max_lines=3,
        )
    draw.text(
        (70, 814),
        "PAIR-LOCKED  /  RANDOM-CONTROLLED  /  RUNTIME-HASHED",
        font=font(27, True),
        fill=INK,
    )
    draw.text(
        (70, 882),
        "100 fixed pairs per cell  |  seed 42  |  300 x 30 CEM  |  solver batch 1",
        font=font(23),
        fill=MUTED,
    )
    draw.text(
        (70, 948),
        "Open the task guides for the exact Moderate and Strict definitions.",
        font=font(23, True),
        fill=GREEN,
    )
    return image


def build_overview(
    results: dict,
    rgb_sequences: dict[str, list[Image.Image]],
    source: dict,
    comparison: dict,
    output: Path,
    preview: Path,
    poster: Path,
) -> None:
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
        f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
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

    def emit(frame: Image.Image) -> None:
        nonlocal frame_number
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        if frame_number % 3 == 0:
            preview_frames.append(frame.resize((1280, 720), Image.Resampling.LANCZOS))
        frame_number += 1

    for _ in range(2 * fps):
        emit(intro_frame())
    for task in TASKS:
        for index in range(int(5.5 * fps)):
            scene_progress = index / (int(5.5 * fps) - 1)
            progress = min(scene_progress / 0.8, 1.0)
            rgb = None
            if task != "tworoom":
                frames = rgb_sequences[task]
                end_index = source[task].get("comparison_end_index", len(frames) - 1)
                frame_index = min(int(progress * end_index), len(frames) - 1)
                rgb = frames[frame_index]
            emit(comparison_frame(task, rgb, source, comparison, progress))
    final = summary_frame(results)
    for _ in range(3 * fps):
        emit(final)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while building overview")
    final.save(poster, optimize=True)
    save_gif(preview_frames, preview, duration=100, colors=144)


def _select_tworoom_trace(payload: dict) -> dict:
    successes = payload["episode_successes"]
    records = payload["topology"]["episodes"]
    candidates = [
        record
        for success, record in zip(successes, records, strict=True)
        if success and record["route_valid"] and record["valid_room_crossings"] == 1
    ]
    if not candidates:
        raise RuntimeError("No legal successful TwoRoom trace found")
    return min(candidates, key=lambda record: record["blocked_steps"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(TRACE_SOURCE.read_text())
    comparison = json.loads(TWOROOM_COMPARISON.read_text())
    tworoom_payload = json.loads(TWOROOM_RESULT.read_text())
    tworoom_trace = _select_tworoom_trace(tworoom_payload)
    results = {
        task: {
            "moderate": source["results"][task]["moderate"],
            "strict": (
                source["results"][task]["strict_swept"]
                if task == "tworoom"
                else source["results"][task]["strict"]
            ),
        }
        for task in TASKS
    }
    episode_indices = {"pusht": 4, "cube": 105, "reacher": 4324}
    data_keys = {
        "pusht": ("state",),
        "cube": (
            "privileged_block_0_pos",
            "privileged_target_block_pos",
            "privileged_block_0_quat",
            "privileged_target_block_yaw",
        ),
        "reacher": ("qpos",),
    }
    episode_payloads = {
        task: load_episode_data(
            args.cache_dir / DATASETS[task],
            episode_indices[task],
            135,
            data_keys[task],
        )
        for task in ("pusht", "cube", "reacher")
    }
    rgb_sequences = {
        task: episode_payloads[task]["frames"] for task in ("pusht", "cube", "reacher")
    }
    task_sources = {
        task: synchronized_metric_trace(task, episode_payloads[task])
        for task in ("pusht", "cube", "reacher")
    } | {
        "tworoom": tworoom_trace,
    }
    for task in TASKS:
        frames = []
        for index in range(72):
            source_index = int(round(index / 71 * 134))
            rgb = None if task == "tworoom" else rgb_sequences[task][source_index]
            frames.append(
                task_frame(
                    task,
                    rgb,
                    task_sources[task],
                    results[task],
                    index / 71,
                )
            )
        save_gif(
            frames,
            ASSETS / "task_gifs" / f"{task}.gif",
            duration=67,
            colors=192,
        )
    showcase = ASSETS / "showcase"
    showcase.mkdir(parents=True, exist_ok=True)
    build_overview(
        results,
        rgb_sequences,
        task_sources,
        comparison,
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
