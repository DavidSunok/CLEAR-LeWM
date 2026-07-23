from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product

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


def wrapped_angle_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest absolute error for periodic angles in radians."""
    delta = np.asarray(current, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def reacher_joint_error(
    current: np.ndarray, target: np.ndarray, mode: str
) -> np.ndarray:
    """Return errors under the joint topology used by DMC Reacher."""
    raw = np.abs(
        np.asarray(current, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    )
    if mode == "raw":
        return raw
    wrapped = wrapped_angle_error(current, target)
    if mode == "all-periodic":
        return wrapped
    if mode == "shoulder-periodic":
        result = raw.copy()
        result[..., 0] = wrapped[..., 0]
        return result
    raise ValueError(f"Unknown Reacher angle mode: {mode}")


def _cube_symmetry_matrices() -> np.ndarray:
    matrices = []
    identity = np.eye(3, dtype=np.float64)
    for permutation in permutations(range(3)):
        base = identity[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = base * np.asarray(signs, dtype=np.float64)[None, :]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    result = np.asarray(matrices, dtype=np.float64)
    if result.shape != (24, 3, 3):
        raise RuntimeError(f"Expected 24 cube symmetries, got {result.shape}")
    return result


CUBE_SYMMETRY_MATRICES = _cube_symmetry_matrices()


def _quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.clip(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12, None
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((*quaternion.shape[:-1], 3, 3))


def cube_symmetry_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Geodesic orientation error modulo the cube's 24 proper rotations."""
    current = _quaternion_matrix_wxyz(q0)
    target = _quaternion_matrix_wxyz(q1)
    relative = np.swapaxes(current, -1, -2) @ target
    # trace(relative @ symmetry) without materializing an N x 24 x 3 x 3 tensor.
    traces = np.einsum(
        "...ij,kji->...k", relative, CUBE_SYMMETRY_MATRICES, optimize=True
    )
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(np.min(angles, axis=-1))


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
            extras["symmetry_orientation_distance_deg"] = cube_symmetry_angle_deg(
                start_q, goal_q
            )
        return PairDiagnostics(
            difficulty=position_m,
            initial_success=position_m <= 0.04,
            extras=extras,
        )

    if task == "pusht":
        start, goal = _read_pair(dataset, "state", starts, goals)
        upstream_position = np.linalg.norm(goal[:, :4] - start[:, :4], axis=1)
        block_position = np.linalg.norm(goal[:, 2:4] - start[:, 2:4], axis=1)
        angle = np.abs(goal[:, 4] - start[:, 4])
        angle = np.minimum(angle, 2.0 * np.pi - angle)
        success = (upstream_position < 20.0) & (angle < np.pi / 9.0)
        return PairDiagnostics(
            difficulty=block_position,
            initial_success=success,
            extras={
                "position_distance": upstream_position,
                "block_position_distance": block_position,
                "angle_distance_deg": np.degrees(angle),
            },
        )

    if task == "reacher":
        start, goal = _read_pair(dataset, "qpos", starts, goals)
        start_finger, goal_finger = _read_pair(dataset, "finger_pos", starts, goals)
        endpoint_distance = np.linalg.norm(goal_finger - start_finger, axis=1)
        raw_max_joint = np.max(np.abs(goal - start), axis=1)
        wrapped_max_joint = np.max(wrapped_angle_error(goal, start), axis=1)
        topology_max_joint = np.max(
            reacher_joint_error(goal, start, "shoulder-periodic"), axis=1
        )
        return PairDiagnostics(
            difficulty=topology_max_joint,
            initial_success=raw_max_joint < 0.05,
            extras={
                "max_joint_distance_rad": raw_max_joint,
                "max_wrapped_joint_distance_rad": wrapped_max_joint,
                "max_topology_joint_distance_rad": topology_max_joint,
                "endpoint_distance_m": endpoint_distance,
            },
        )

    start, goal = _read_pair(dataset, "proprio", starts, goals)
    distance = np.linalg.norm(goal - start, axis=1)
    from .topology import canonical_tworoom_geometry, point_is_clear

    geometry = canonical_tworoom_geometry()
    cross_room = np.asarray(
        [
            geometry.is_cross_room(current, target)
            for current, target in zip(start, goal)
        ]
    )
    start_clear = np.asarray([point_is_clear(geometry, point) for point in start])
    goal_clear = np.asarray([point_is_clear(geometry, point) for point in goal])
    geodesic = np.asarray(
        [
            geometry.shortest_door_path(current, target)
            for current, target in zip(start, goal)
        ]
    )
    return PairDiagnostics(
        difficulty=geodesic,
        initial_success=distance < 16.0,
        extras={
            "euclidean_distance": distance,
            "geodesic_distance": geodesic,
            "cross_room": cross_room,
            "start_clear": start_clear,
            "goal_clear": goal_clear,
        },
    )


def tworoom_invalid_transition_prefix(dataset: h5py.File) -> np.ndarray:
    """Prefix count of physically invalid transitions in a TwoRoom dataset."""
    from .datasets import episode_column
    from .topology import canonical_tworoom_geometry, check_route_segment

    positions = np.asarray(dataset["proprio"][:], dtype=np.float64)
    episodes = np.asarray(dataset[episode_column(dataset)][:])
    steps = np.asarray(dataset["step_idx"][:])
    if len(positions) < 2:
        return np.zeros(len(positions), dtype=np.int64)

    adjacent = (episodes[1:] == episodes[:-1]) & (steps[1:] == steps[:-1] + 1)
    starts = positions[:-1]
    ends = positions[1:]
    segment_min = np.minimum(starts, ends)
    segment_max = np.maximum(starts, ends)
    geometry = canonical_tworoom_geometry()

    candidates = ~adjacent
    low, high = geometry.center_bounds
    candidates |= np.any((starts < low) | (starts > high), axis=1)
    candidates |= np.any((ends < low) | (ends > high), axis=1)
    radius = geometry.agent_radius
    for rectangle in geometry.solid_wall_rectangles():
        lower = np.asarray(
            [rectangle.xmin - radius, rectangle.ymin - radius], dtype=np.float64
        )
        upper = np.asarray(
            [rectangle.xmax + radius, rectangle.ymax + radius], dtype=np.float64
        )
        candidates |= np.all(segment_max >= lower, axis=1) & np.all(
            segment_min <= upper, axis=1
        )

    invalid = ~adjacent
    for index in np.flatnonzero(candidates & adjacent):
        invalid[index] = not check_route_segment(
            geometry, starts[index], ends[index]
        ).valid
    return np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(invalid, dtype=np.int64))
    )


def tworoom_source_window_clean(
    invalid_prefix: np.ndarray, starts: np.ndarray, goals: np.ndarray
) -> np.ndarray:
    """Return whether every transition in each [start, goal] window is legal."""
    starts = np.asarray(starts, dtype=np.int64)
    goals = np.asarray(goals, dtype=np.int64)
    return (invalid_prefix[goals] - invalid_prefix[starts]) == 0
