from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .datasets import (
    episode_column,
    file_sha256,
    metadata_fingerprint,
    sample_rows,
    split_episode_ids,
    valid_pairs,
)
from .protocols import ProtocolSpec, get_protocol, normalize_task
from .tasks import pair_diagnostics

SCHEMA_VERSION = "clear-lewm-manifest-v1"


def _criterion_initial_success(diagnostics, task: str, spec: ProtocolSpec):
    if spec.success_mode != "task-sustained":
        return diagnostics.initial_success
    extras = diagnostics.extras
    if task == "cube":
        position_ok = extras["position_distance_m"] <= spec.cube_position_threshold_m
        if spec.cube_orientation_threshold_deg is None:
            return position_ok
        orientation_key = (
            "symmetry_orientation_distance_deg"
            if spec.cube_symmetry_aware
            else "orientation_distance_deg"
        )
        return position_ok & (
            extras[orientation_key] <= spec.cube_orientation_threshold_deg
        )
    if task == "pusht":
        position_key = (
            "block_position_distance" if spec.pusht_block_only else "position_distance"
        )
        return (extras[position_key] < spec.pusht_position_threshold) & (
            extras["angle_distance_deg"] < spec.pusht_angle_threshold_deg
        )
    if task == "reacher":
        distance_key = (
            "max_wrapped_joint_distance_rad"
            if spec.reacher_wrap_angles
            else "max_joint_distance_rad"
        )
        return extras[distance_key] < spec.reacher_joint_threshold_rad
    return extras["euclidean_distance"] < spec.tworoom_distance_threshold


def _criterion_difficulty(diagnostics, task: str, spec: ProtocolSpec):
    extras = diagnostics.extras
    if task == "pusht" and not spec.pusht_block_only:
        return extras["position_distance"]
    if task == "reacher" and not spec.reacher_wrap_angles:
        return extras["max_joint_distance_rad"]
    if task == "tworoom" and not spec.tworoom_route_required:
        return extras["euclidean_distance"]
    return diagnostics.difficulty


def _json_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def generate_manifest(
    dataset_path: str | Path,
    task: str,
    protocol: str | ProtocolSpec,
    num_eval: int,
    seed: int,
    split: str | None = None,
    full_sha256: bool = False,
) -> dict:
    dataset_path = Path(dataset_path).resolve()
    task = normalize_task(task)
    spec = get_protocol(protocol) if isinstance(protocol, str) else protocol
    selected_split = split or spec.split
    if selected_split not in {"all", "train", "heldout"}:
        raise ValueError("split must be one of: all, train, heldout")

    with h5py.File(dataset_path, "r") as dataset:
        starts, goals, episodes, steps = valid_pairs(dataset, spec.goal_offset)
        valid_count = len(starts)
        train_ids, heldout_ids = split_episode_ids(
            episodes, spec.heldout_fraction, seed
        )
        if selected_split == "train":
            split_mask = np.isin(episodes[starts], train_ids)
        elif selected_split == "heldout":
            if len(heldout_ids) == 0:
                raise ValueError(
                    f"Protocol {spec.name} does not define a held-out split"
                )
            split_mask = np.isin(episodes[starts], heldout_ids)
        else:
            split_mask = np.ones(len(starts), dtype=bool)
        starts = starts[split_mask]
        goals = goals[split_mask]

        diagnostics = pair_diagnostics(dataset, task, starts, goals)
        criterion_difficulty = _criterion_difficulty(diagnostics, task, spec)
        criterion_initial = _criterion_initial_success(diagnostics, task, spec)
        initial_success_rate = float(criterion_initial.mean() * 100.0)
        upstream_initial_success_rate = float(
            diagnostics.initial_success.mean() * 100.0
        )
        keep = np.ones(len(starts), dtype=bool)
        if spec.exclude_initial_success:
            keep &= ~criterion_initial
        if task == "tworoom" and spec.tworoom_crossroom_only:
            keep &= diagnostics.extras["cross_room"]
            keep &= diagnostics.extras["start_clear"]
            keep &= diagnostics.extras["goal_clear"]
        min_difficulty = spec.min_difficulty.get(task)
        if min_difficulty is not None:
            keep &= criterion_difficulty >= min_difficulty
        filtered_starts = starts[keep]
        sampling_candidates = filtered_starts
        if spec.reproduce_upstream_off_by_one:
            sampling_candidates = sampling_candidates[:-1]

        chosen = sample_rows(
            sampling_candidates,
            episodes,
            num_eval=num_eval,
            seed=seed,
            mode=spec.sampling,
        )
        chosen_goals = chosen + spec.goal_offset
        chosen_diag = pair_diagnostics(dataset, task, chosen, chosen_goals)
        chosen_difficulty = _criterion_difficulty(chosen_diag, task, spec)
        chosen_initial = _criterion_initial_success(chosen_diag, task, spec)
        ep_key = episode_column(dataset)

        pairs = []
        for index, (start_row, goal_row) in enumerate(zip(chosen, chosen_goals)):
            pair = {
                "pair_id": index,
                "episode_id": _json_scalar(episodes[start_row]),
                "start_step": int(steps[start_row]),
                "goal_step": int(steps[goal_row]),
                "start_row": int(start_row),
                "goal_row": int(goal_row),
                "difficulty": float(chosen_difficulty[index]),
                "initial_success": bool(chosen_initial[index]),
                "upstream_initial_success": bool(chosen_diag.initial_success[index]),
            }
            for name, values in chosen_diag.extras.items():
                pair[name] = float(values[index])
            pairs.append(pair)

    fingerprint = {
        "kind": "metadata-sha256",
        "value": metadata_fingerprint(dataset_path),
    }
    if full_sha256:
        fingerprint = {"kind": "file-sha256", "value": file_sha256(dataset_path)}

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "protocol": spec.to_dict(),
        "seed": seed,
        "policy_seed": seed,
        "split": selected_split,
        "dataset": {
            "name": dataset_path.name,
            "episode_column": ep_key,
            "fingerprint": fingerprint,
        },
        "statistics": {
            "valid_pairs_all_splits": valid_count,
            "pairs_in_selected_split": int(len(starts)),
            "initial_success_rate_percent": initial_success_rate,
            "upstream_initial_success_rate_percent": (upstream_initial_success_rate),
            "pairs_after_filters": int(len(filtered_starts)),
            "pairs_available_to_sampler": int(len(sampling_candidates)),
            "num_eval": num_eval,
            "unique_episodes": int(len(np.unique(episodes[chosen]))),
        },
        "pairs": pairs,
    }


def save_manifest(manifest: dict, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def load_manifest(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {data.get('schema_version')!r}")
    return data
