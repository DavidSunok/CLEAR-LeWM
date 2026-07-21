#!/usr/bin/env python3
"""Build the README protocol animation from canonical dataset RGB frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers compressed HDF5 filters
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TASK_FILES = {
    "PushT": "pusht_expert_train.h5",
    "Cube": "datasets/ogbench/cube_single_expert.h5",
    "Reacher": "datasets/dmc/reacher_random.h5",
    "TwoRoom": "datasets/tworoom.h5",
}

TIERS = (
    (
        "OFFICIAL",
        "Exact upstream compatibility: retained starts and first-hit success",
        "#667085",
        {
            "PushT": ("pos < 20; angle < 20 deg", "first hit"),
            "Cube": ("position <= 4 cm", "first hit; orientation ignored"),
            "Reacher": ("joint error < 0.05 rad", "first hit"),
            "TwoRoom": ("distance < 16", "first hit"),
        },
    ),
    (
        "MODERATE",
        "Balanced non-trivial goals with stable target attainment",
        "#d97706",
        {
            "PushT": ("pos < 20; angle < 20 deg", "hold 3 steps"),
            "Cube": ("pos <= 4 cm; ori <= 30 deg", "hold 3 steps"),
            "Reacher": ("joint error < 0.10 rad", "hold 2 steps"),
            "TwoRoom": ("distance < 12", "hold 3 steps"),
        },
    ),
    (
        "STRICT",
        "Tighter geometry, sustained success, and larger start-goal change",
        "#15803d",
        {
            "PushT": ("pos < 15; angle < 15 deg", "hold 3 steps"),
            "Cube": ("pos <= 3 cm; ori <= 15 deg", "hold 5 steps"),
            "Reacher": ("joint error < 0.075 rad", "hold 2 steps"),
            "TwoRoom": ("distance < 8", "hold 5 steps"),
        },
    ),
)


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    return ImageFont.truetype(str(path), size=size)


def _load_frames(path: Path, count: int, episode_index: int) -> list[Image.Image]:
    with h5py.File(path, "r") as dataset:
        steps = np.asarray(dataset["step_idx"])
        starts = np.flatnonzero(steps == 0)
        start = int(starts[min(episode_index, len(starts) - 1)])
        episode_end = (
            int(starts[episode_index + 1])
            if episode_index + 1 < len(starts)
            else len(steps)
        )
        offsets = np.linspace(0, max(episode_end - start - 1, 0), count).astype(int)
        arrays = np.asarray(dataset["pixels"][start + offsets])
    return [Image.fromarray(frame).resize((184, 184)) for frame in arrays]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets/protocols.gif"))
    parser.add_argument("--frames-per-tier", type=int, default=12)
    args = parser.parse_args()

    tasks = list(TASK_FILES)
    task_frames = {
        task: _load_frames(
            args.cache_dir / relative,
            args.frames_per_tier,
            episode_index=3 + index,
        )
        for index, (task, relative) in enumerate(TASK_FILES.items())
    }
    width, height = 848, 314
    frames = []
    title_font = _font(21, bold=True)
    subtitle_font = _font(13)
    task_font = _font(15, bold=True)
    criterion_font = _font(10)

    for tier_name, subtitle, color, criteria in TIERS:
        for frame_index in range(args.frames_per_tier):
            canvas = Image.new("RGB", (width, height), "#f8fafc")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, width, 48), fill="#101828")
            draw.text((14, 8), tier_name, fill=color, font=title_font)
            draw.text((175, 13), subtitle, fill="#f2f4f7", font=subtitle_font)
            for task_index, task in enumerate(tasks):
                x = 8 + task_index * 210
                draw.rectangle(
                    (x, 56, x + 202, 306), fill="#ffffff", outline=color, width=3
                )
                draw.text((x + 9, 63), task, fill="#101828", font=task_font)
                canvas.paste(task_frames[task][frame_index], (x + 9, 86))
                first, second = criteria[task]
                draw.text((x + 9, 274), first, fill="#344054", font=criterion_font)
                draw.text((x + 9, 286), second, fill=color, font=criterion_font)
            frames.append(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
