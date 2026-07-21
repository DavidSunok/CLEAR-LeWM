from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np

from .protocols import normalize_task


@dataclass
class PairDiagnostics:
    difficulty: np.ndarray
    initial_success: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)


def quaternion_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.clip(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-12, None)
    q1 = q1 / np.clip(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12, None)
    dot = np.abs(np.sum(q0 * q1, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _read_pair(
    dataset: h5py.File, key: str, starts: np.ndarray, goals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if key not in dataset:
        raise KeyError(f"Dataset is missing required column {key!r}")
    return np.asarray(dataset[key][starts]), np.asarray(dataset[key][goals])


def pair_diagnostics(
    dataset: h5py.File,
    task: str,
    starts: np.ndarray,
    goals: np.ndarray,
) -> PairDiagnostics:
    task = normalize_task(task)
    starts = np.asarray(starts, dtype=np.int64)
    goals = np.asarray(goals, dtype=np.int64)

    if task == "cube":
        start_pos, goal_pos = _read_pair(
            dataset, "privileged_block_0_pos", starts, goals
        )
        position_m = np.linalg.norm(goal_pos - start_pos, axis=1)
        extras: dict[str, np.ndarray] = {"position_distance_m": position_m}
        if "privileged_block_0_quat" in dataset:
            start_q, goal_q = _read_pair(
                dataset, "privileged_block_0_quat", starts, goals
            )
            extras["orientation_distance_deg"] = quaternion_angle_deg(start_q, goal_q)
        return PairDiagnostics(
            difficulty=position_m,
            initial_success=position_m <= 0.04,
            extras=extras,
        )

    if task == "pusht":
        start, goal = _read_pair(dataset, "state", starts, goals)
        position = np.linalg.norm(goal[:, :4] - start[:, :4], axis=1)
        angle = np.abs(goal[:, 4] - start[:, 4])
        angle = np.minimum(angle, 2.0 * np.pi - angle)
        success = (position < 20.0) & (angle < np.pi / 9.0)
        return PairDiagnostics(
            difficulty=position,
            initial_success=success,
            extras={
                "position_distance": position,
                "angle_distance_deg": np.degrees(angle),
            },
        )

    if task == "reacher":
        start, goal = _read_pair(dataset, "qpos", starts, goals)
        max_joint = np.max(np.abs(goal - start), axis=1)
        return PairDiagnostics(
            difficulty=max_joint,
            initial_success=max_joint < 0.05,
            extras={"max_joint_distance_rad": max_joint},
        )

    start, goal = _read_pair(dataset, "proprio", starts, goals)
    distance = np.linalg.norm(goal - start, axis=1)
    return PairDiagnostics(
        difficulty=distance,
        initial_success=distance < 16.0,
        extras={"euclidean_distance": distance},
    )
