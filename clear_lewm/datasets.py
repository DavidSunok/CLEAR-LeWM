from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def episode_column(dataset: h5py.File) -> str:
    for name in ("episode_idx", "ep_idx"):
        if name in dataset:
            return name
    raise KeyError("Dataset must contain 'episode_idx' or 'ep_idx'")


def valid_pairs(
    dataset: h5py.File, goal_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if goal_offset <= 0:
        raise ValueError("goal_offset must be positive")
    ep_key = episode_column(dataset)
    episodes = np.asarray(dataset[ep_key][:])
    steps = np.asarray(dataset["step_idx"][:])
    rows = np.arange(len(episodes), dtype=np.int64)
    goal_rows = rows + goal_offset
    in_bounds = goal_rows < len(rows)
    safe_goals = np.minimum(goal_rows, len(rows) - 1)
    contiguous = (
        in_bounds
        & (episodes[safe_goals] == episodes)
        & (steps[safe_goals] == steps + goal_offset)
    )
    starts = rows[contiguous]
    return starts, starts + goal_offset, episodes, steps


def split_episode_ids(
    episode_ids: np.ndarray, heldout_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(episode_ids)
    if heldout_fraction <= 0:
        return unique, np.empty(0, dtype=unique.dtype)
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in [0, 1)")

    scores = []
    for episode in unique:
        payload = f"clear-lewm:{seed}:{episode}".encode()
        scores.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
    order = np.argsort(np.asarray(scores, dtype=np.uint64), kind="stable")
    n_heldout = max(1, int(round(len(unique) * heldout_fraction)))
    heldout = np.sort(unique[order[:n_heldout]])
    train = np.sort(unique[order[n_heldout:]])
    return train, heldout


def sample_rows(
    candidates: np.ndarray,
    episode_ids: np.ndarray,
    num_eval: int,
    seed: int,
    mode: str,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if num_eval <= 0:
        raise ValueError("num_eval must be positive")
    if len(candidates) < num_eval:
        raise ValueError(
            f"Requested {num_eval} pairs, but only {len(candidates)} are available"
        )

    rng = np.random.default_rng(seed)
    if mode == "row-uniform":
        return np.sort(rng.choice(candidates, size=num_eval, replace=False))
    if mode != "episode-balanced":
        raise ValueError(f"Unknown sampling mode: {mode}")

    candidate_episodes = episode_ids[candidates]
    unique, first, counts = np.unique(
        candidate_episodes, return_index=True, return_counts=True
    )
    if num_eval <= len(unique):
        selected_groups = rng.choice(len(unique), size=num_eval, replace=False)
    else:
        cycles, remainder = divmod(num_eval, len(unique))
        selected_groups = np.concatenate(
            [
                np.concatenate([rng.permutation(len(unique)) for _ in range(cycles)]),
                rng.choice(len(unique), size=remainder, replace=False),
            ]
        )

    selected = []
    used: set[int] = set()
    for group in selected_groups:
        group_rows = candidates[first[group] : first[group] + counts[group]]
        available = np.asarray([row for row in group_rows if int(row) not in used])
        if len(available) == 0:
            available = group_rows
        row = int(rng.choice(available))
        selected.append(row)
        used.add(row)
    return np.sort(np.asarray(selected, dtype=np.int64))


def metadata_fingerprint(path: str | Path) -> str:
    path = Path(path)
    with h5py.File(path, "r") as dataset:
        schema = []
        for key in sorted(dataset.keys()):
            obj = dataset[key]
            if isinstance(obj, h5py.Dataset):
                schema.append(
                    {"key": key, "shape": list(obj.shape), "dtype": str(obj.dtype)}
                )
    payload = {
        "size_bytes": path.stat().st_size,
        "schema": schema,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
