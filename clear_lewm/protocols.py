from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field

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
    pusht_position_threshold: float = 20.0
    pusht_angle_threshold_deg: float = 20.0
    pusht_block_only: bool = False
    reacher_joint_threshold_rad: float = 0.05
    reacher_wrap_angles: bool = False
    reacher_angle_mode: str | None = None
    reacher_success_mode: str = "joint"
    reacher_endpoint_threshold_m: float = 0.015
    tworoom_distance_threshold: float = 16.0
    tworoom_crossroom_only: bool = False
    tworoom_source_window_clean: bool = False
    tworoom_route_required: bool = False
    tworoom_goal_side_required: bool = False
    tworoom_collision_mode: str = "official"
    cube_symmetry_aware: bool = False
    sustained_steps: int = 1
    pusht_sustained_steps: int | None = None
    cube_sustained_steps: int | None = None
    reacher_sustained_steps: int | None = None
    tworoom_sustained_steps: int | None = None
    success_mode: str = "upstream"
    reproduce_upstream_off_by_one: bool = False
    min_difficulty: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def hold_steps(self, task: str) -> int:
        task_value = getattr(self, f"{task}_sustained_steps")
        return int(task_value if task_value is not None else self.sustained_steps)

    def resolved_reacher_angle_mode(self) -> str:
        if self.reacher_angle_mode is not None:
            return self.reacher_angle_mode
        return "all-periodic" if self.reacher_wrap_angles else "raw"


PROTOCOLS: dict[str, ProtocolSpec] = {
    "official": ProtocolSpec(
        name="official",
        description=(
            "Exact LeWM-compatible evaluation: row-uniform sampling over all "
            "valid rows, initially solved pairs retained, and upstream first-hit "
            "success predicates."
        ),
        sampling="row-uniform",
        split="all",
        heldout_fraction=0.0,
        exclude_initial_success=False,
        reproduce_upstream_off_by_one=True,
    ),
    "moderate": ProtocolSpec(
        name="moderate",
        description=(
            "Minimal LeWM-compatible correction: remove initially solved pairs, "
            "repair Reacher joint topology and TwoRoom collision/data defects, "
            "and otherwise preserve the released task predicates."
        ),
        sampling="episode-balanced",
        split="all",
        heldout_fraction=0.0,
        exclude_initial_success=True,
        cube_orientation_threshold_deg=None,
        cube_symmetry_aware=False,
        pusht_block_only=False,
        reacher_joint_threshold_rad=0.05,
        reacher_angle_mode="shoulder-periodic",
        tworoom_distance_threshold=16.0,
        tworoom_crossroom_only=True,
        tworoom_source_window_clean=True,
        tworoom_route_required=False,
        tworoom_collision_mode="swept",
        sustained_steps=1,
        success_mode="task-sustained",
        min_difficulty={},
    ),
    "strict": ProtocolSpec(
        name="strict",
        description=(
            "Task-semantic completion built on Moderate data hygiene and physics, "
            "with tighter object, endpoint, persistence, and topology criteria."
        ),
        sampling="episode-balanced",
        split="all",
        heldout_fraction=0.0,
        exclude_initial_success=True,
        cube_position_threshold_m=0.03,
        cube_orientation_threshold_deg=15.0,
        cube_symmetry_aware=True,
        pusht_position_threshold=10.0,
        pusht_angle_threshold_deg=10.0,
        pusht_block_only=True,
        reacher_success_mode="endpoint",
        reacher_endpoint_threshold_m=0.01,
        tworoom_distance_threshold=8.0,
        tworoom_crossroom_only=True,
        tworoom_source_window_clean=True,
        tworoom_route_required=True,
        tworoom_goal_side_required=True,
        tworoom_collision_mode="swept",
        sustained_steps=1,
        pusht_sustained_steps=3,
        cube_sustained_steps=3,
        reacher_sustained_steps=2,
        success_mode="task-sustained",
        min_difficulty={},
    ),
    # v0.1 names remain registered so existing manifests stay executable.
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
        success_mode="cube-pose",
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
        success_mode="cube-pose",
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
        success_mode="cube-pose",
        min_difficulty={
            "cube": 0.08,
            "pusht": 50.0,
            "reacher": 0.25,
            "tworoom": 32.0,
        },
    ),
}

PRIMARY_PROTOCOLS = ("official", "moderate", "strict")


def get_protocol(name: str) -> ProtocolSpec:
    try:
        return PROTOCOLS[name]
    except KeyError as exc:
        choices = ", ".join(PROTOCOLS)
        raise ValueError(
            f"Unknown protocol {name!r}. Choose one of: {choices}"
        ) from exc


def protocol_from_dict(data: dict) -> ProtocolSpec:
    """Restore the exact protocol embedded in a versioned manifest."""
    get_protocol(data["name"])
    values = {}
    for name, definition in ProtocolSpec.__dataclass_fields__.items():
        if name in data:
            values[name] = data[name]
        elif definition.default is not MISSING:
            values[name] = definition.default
        elif definition.default_factory is not MISSING:
            values[name] = definition.default_factory()
        else:
            raise ValueError(f"Manifest protocol is missing required field {name!r}")
    return ProtocolSpec(**values)


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
