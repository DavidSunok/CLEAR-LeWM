"""Route-valid TwoRoom runtime hooks for Moderate and Strict evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType

import numpy as np

from .topology import (
    TwoRoomGeometry,
    check_route_segment,
    point_is_clear,
    resolve_motion,
)


@dataclass
class EpisodeAudit:
    env_index: int
    route_valid: bool = True
    collision_contacts: int = 0
    blocked_steps: int = 0
    valid_room_crossings: int = 0
    hold_count: int = 0
    initial_state_clear: bool = True
    goal_state_clear: bool = True
    cross_room_goal: bool = False
    positions: list[list[float]] = field(default_factory=list)
    goal_position: list[float] | None = None
    geometry: dict = field(default_factory=dict)

    def reset(self, position, geometry: TwoRoomGeometry) -> None:
        self.route_valid = point_is_clear(geometry, position)
        self.collision_contacts = 0
        self.blocked_steps = 0
        self.valid_room_crossings = 0
        self.hold_count = 0
        self.initial_state_clear = self.route_valid
        self.goal_state_clear = True
        self.cross_room_goal = False
        self.positions = [np.asarray(position, dtype=np.float64).tolist()]
        self.goal_position = None
        self.geometry = geometry.to_dict()

    def to_dict(self) -> dict:
        return {
            "env_index": self.env_index,
            "route_valid": self.route_valid,
            "collision_contacts": self.collision_contacts,
            "blocked_steps": self.blocked_steps,
            "valid_room_crossings": self.valid_room_crossings,
            "hold_count": self.hold_count,
            "initial_state_clear": self.initial_state_clear,
            "goal_state_clear": self.goal_state_clear,
            "cross_room_goal": self.cross_room_goal,
            "positions": self.positions,
            "goal_position": self.goal_position,
            "geometry": self.geometry,
        }


ACTIVE_AUDITS: list[EpisodeAudit] = []


def install_topology_success(world, protocol) -> None:
    """Install swept-disk collision and route-gated success on every env."""
    if protocol.tworoom_collision_mode != "swept":
        raise ValueError(
            "Route-required TwoRoom currently supports only swept collision"
        )
    ACTIVE_AUDITS.clear()
    for env_index, wrapped in enumerate(world.envs.envs):
        env = wrapped.unwrapped
        audit = EpisodeAudit(env_index=env_index)
        ACTIVE_AUDITS.append(audit)
        original_set_state = env._set_state
        original_set_goal = env._set_goal_state
        original_step = env.step

        def apply_collisions(self, position, desired, *, _audit=audit):
            geometry = TwoRoomGeometry.from_env(self)
            motion = resolve_motion(geometry, position, desired)
            _audit.collision_contacts += motion.collisions
            _audit.blocked_steps += int(motion.blocked)
            import torch

            return torch.as_tensor(
                motion.position,
                dtype=getattr(position, "dtype", torch.float32),
                device=getattr(position, "device", None),
            )

        def set_state(
            self, state, *, _audit=audit, _original_set_state=original_set_state
        ):
            result = _original_set_state(state)
            _audit.reset(self.agent_position, TwoRoomGeometry.from_env(self))
            self._clear_lewm_route_valid = _audit.route_valid
            self._clear_lewm_hold_count = 0
            return result

        def set_goal_state(
            self,
            goal_state,
            *,
            _audit=audit,
            _original_set_goal=original_set_goal,
        ):
            result = _original_set_goal(goal_state)
            geometry = TwoRoomGeometry.from_env(self)
            _audit.goal_position = np.asarray(goal_state, dtype=np.float64).tolist()
            _audit.goal_state_clear = point_is_clear(geometry, goal_state)
            _audit.cross_room_goal = geometry.is_cross_room(
                self.agent_position, goal_state
            )
            _audit.route_valid &= _audit.goal_state_clear
            _audit.hold_count = 0
            self._clear_lewm_route_valid = _audit.route_valid
            self._clear_lewm_hold_count = 0
            return result

        def step(self, action, *, _audit=audit, _original_step=original_step):
            before = np.asarray(self.agent_position, dtype=np.float64).copy()
            observation, reward, _, truncated, info = _original_step(action)
            after = np.asarray(self.agent_position, dtype=np.float64).copy()
            geometry = TwoRoomGeometry.from_env(self)
            route_check = check_route_segment(geometry, before, after)
            coordinate = geometry.crossing_coordinate(before, after)
            if coordinate is not None and geometry.crossing_has_full_clearance(
                coordinate
            ):
                _audit.valid_room_crossings += 1
            _audit.route_valid &= route_check.valid
            _audit.positions.append(after.tolist())
            self._clear_lewm_route_valid = _audit.route_valid

            distance = float(
                np.linalg.norm(
                    np.asarray(self.agent_position) - np.asarray(self.target_position)
                )
            )
            crossing_ok = not _audit.cross_room_goal or _audit.valid_room_crossings > 0
            success = bool(
                distance < protocol.tworoom_distance_threshold
                and _audit.route_valid
                and crossing_ok
            )
            _audit.hold_count = _audit.hold_count + 1 if success else 0
            self._clear_lewm_hold_count = _audit.hold_count
            terminated = _audit.hold_count >= protocol.hold_steps("tworoom")
            info["clear_lewm_route_valid"] = _audit.route_valid
            info["clear_lewm_valid_room_crossings"] = _audit.valid_room_crossings
            info["clear_lewm_hold_count"] = _audit.hold_count
            return observation, reward, terminated, truncated, info

        env._apply_collisions = MethodType(apply_collisions, env)
        env._set_state = MethodType(set_state, env)
        env._set_goal_state = MethodType(set_goal_state, env)
        env.step = MethodType(step, env)


def topology_audit_records() -> list[dict]:
    return [audit.to_dict() for audit in ACTIVE_AUDITS]
