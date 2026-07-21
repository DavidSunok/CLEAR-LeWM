from __future__ import annotations

from dataclasses import asdict, dataclass, field

TASKS = ("pusht", "reacher", "tworoom", "cube")


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    description: str
    sampling: str
    split: str
    heldout_fraction: float
    exclude_initial_success: bool
    goal_offset: int = 25
    eval_budget: int = 50
    cube_position_threshold_m: float = 0.04
    cube_orientation_threshold_deg: float | None = None
    sustained_steps: int = 1
    reproduce_upstream_off_by_one: bool = False
    min_difficulty: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


PROTOCOLS: dict[str, ProtocolSpec] = {
    "official-compat": ProtocolSpec(
        name="official-compat",
        description=(
            "LeWM-compatible row-uniform sampling from all episodes. This track "
            "exists only for comparison with previously reported numbers."
        ),
        sampling="row-uniform",
        split="all",
        heldout_fraction=0.0,
        exclude_initial_success=False,
        reproduce_upstream_off_by_one=True,
    ),
    "clear-id": ProtocolSpec(
        name="clear-id",
        description=(
            "Difficulty-controlled in-distribution evaluation for released "
            "checkpoints that were trained on the full public dataset."
        ),
        sampling="episode-balanced",
        split="all",
        heldout_fraction=0.0,
        exclude_initial_success=True,
        cube_orientation_threshold_deg=15.0,
        sustained_steps=5,
    ),
    "clear-standard": ProtocolSpec(
        name="clear-standard",
        description=(
            "Held-out, episode-balanced evaluation with initially solved pairs "
            "removed and deterministic policy seeding."
        ),
        sampling="episode-balanced",
        split="heldout",
        heldout_fraction=0.2,
        exclude_initial_success=True,
        cube_orientation_threshold_deg=15.0,
        sustained_steps=5,
    ),
    "clear-hard": ProtocolSpec(
        name="clear-hard",
        description=(
            "The CLEAR standard restricted to larger state changes. Thresholds "
            "are expressed in each environment's native state units."
        ),
        sampling="episode-balanced",
        split="heldout",
        heldout_fraction=0.2,
        exclude_initial_success=True,
        cube_orientation_threshold_deg=15.0,
        sustained_steps=5,
        min_difficulty={
            "cube": 0.08,
            "pusht": 50.0,
            "reacher": 0.25,
            "tworoom": 32.0,
        },
    ),
}


def get_protocol(name: str) -> ProtocolSpec:
    try:
        return PROTOCOLS[name]
    except KeyError as exc:
        choices = ", ".join(PROTOCOLS)
        raise ValueError(
            f"Unknown protocol {name!r}. Choose one of: {choices}"
        ) from exc


def normalize_task(task: str) -> str:
    value = task.lower().replace("-", "").replace("_", "")
    aliases = {
        "pusht": "pusht",
        "reacher": "reacher",
        "tworoom": "tworoom",
        "cube": "cube",
        "ogbenchcube": "cube",
        "ogbcube": "cube",
    }
    if value not in aliases:
        raise ValueError(f"Unknown task {task!r}. Choose one of: {', '.join(TASKS)}")
    return aliases[value]
