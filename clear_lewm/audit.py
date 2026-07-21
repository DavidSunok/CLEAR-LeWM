from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .datasets import episode_column, metadata_fingerprint, valid_pairs
from .protocols import normalize_task
from .tasks import pair_diagnostics


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    result = np.quantile(values, levels)
    return {
        f"q{int(level * 100):02d}": float(value) for level, value in zip(levels, result)
    }


def audit_dataset(dataset_path: str | Path, task: str, goal_offset: int = 25) -> dict:
    dataset_path = Path(dataset_path).resolve()
    task = normalize_task(task)
    with h5py.File(dataset_path, "r") as dataset:
        starts, goals, episodes, _ = valid_pairs(dataset, goal_offset)
        diagnostics = pair_diagnostics(dataset, task, starts, goals)
        extras = {
            name: _quantiles(values) for name, values in diagnostics.extras.items()
        }
        return {
            "task": task,
            "dataset": dataset_path.name,
            "dataset_metadata_sha256": metadata_fingerprint(dataset_path),
            "episode_column": episode_column(dataset),
            "episodes": int(len(np.unique(episodes))),
            "goal_offset": int(goal_offset),
            "valid_pairs": int(len(starts)),
            "initial_success_rate_percent": float(
                diagnostics.initial_success.mean() * 100.0
            ),
            "difficulty_quantiles": _quantiles(diagnostics.difficulty),
            "diagnostic_quantiles": extras,
        }
