"""Canonical LeWM training-input profiles for FAST conversion and audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastTaskProfile:
    task: str
    source: str
    keys_to_load: tuple[str, ...]
    keys_to_cache: tuple[str, ...] = ()
    keys_to_merge: tuple[tuple[str, str], ...] = ()

    @property
    def merge_mapping(self) -> dict[str, str]:
        return dict(self.keys_to_merge)


FAST_TASK_PROFILES = {
    "pusht": FastTaskProfile(
        task="pusht",
        source="pusht_expert_train.lance",
        keys_to_load=("pixels", "action", "proprio", "state"),
        keys_to_cache=("action", "proprio", "state"),
    ),
    "cube": FastTaskProfile(
        task="cube",
        source="ogbench/cube_single_expert.h5",
        keys_to_load=("pixels", "action", "observation"),
        keys_to_cache=("action", "observation"),
        keys_to_merge=(("proprio", "proprio"),),
    ),
    "reacher": FastTaskProfile(
        task="reacher",
        source="dmc/reacher_random.h5",
        keys_to_load=("pixels", "action", "observation"),
        keys_to_cache=("action", "observation"),
    ),
    "tworoom": FastTaskProfile(
        task="tworoom",
        source="tworoom.h5",
        keys_to_load=("pixels", "action", "proprio"),
        keys_to_cache=("action", "proprio"),
    ),
}


def get_fast_profile(task: str) -> FastTaskProfile:
    try:
        return FAST_TASK_PROFILES[task]
    except KeyError as exc:
        choices = ", ".join(sorted(FAST_TASK_PROFILES))
        raise ValueError(f"Unknown FAST task {task!r}; choose from {choices}") from exc


def load_profile_dataset(
    task: str,
    cache_dir: str | Path,
    *,
    num_steps: int = 1,
    frameskip: int = 1,
    transform=None,
):
    import stable_worldmodel as swm

    profile = get_fast_profile(task)
    return swm.data.load_dataset(
        profile.source,
        cache_dir=cache_dir,
        num_steps=num_steps,
        frameskip=frameskip,
        transform=transform,
        keys_to_load=list(profile.keys_to_load),
        keys_to_cache=list(profile.keys_to_cache),
        keys_to_merge=profile.merge_mapping or None,
    )
