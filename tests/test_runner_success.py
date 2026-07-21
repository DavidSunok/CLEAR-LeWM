from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from clear_lewm.protocols import get_protocol
from clear_lewm.runner import (
    _checkpoint_record,
    _install_pusht_success,
    _install_reacher_success,
    _install_tworoom_success,
    _portable_manifest_path,
)


def _world(env):
    wrapped = SimpleNamespace(unwrapped=env)
    return SimpleNamespace(envs=SimpleNamespace(envs=[wrapped]))


class FakePushT:
    def __init__(self):
        self.goal_state = np.zeros(7)

    def _set_goal_state(self, goal_state):
        self.goal_state = np.asarray(goal_state)

    def step(self, action):
        return {"state": self.goal_state.copy()}, 0.0, True, False, {}


class FakeTwoRoom:
    def __init__(self):
        self.agent_position = np.zeros(2)
        self.target_position = np.zeros(2)

    def _set_goal_state(self, goal_state):
        self.target_position = np.asarray(goal_state)

    def step(self, action):
        return np.zeros(2), 0.0, True, False, {}


class FakeReacher:
    def __init__(self):
        physics = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(2)))
        task = SimpleNamespace(target_qpos=np.zeros(2))
        self.env = SimpleNamespace(physics=physics, task=task)

    def _is_terminated(self, step):
        return True

    def set_target_qpos(self, target_qpos):
        self.env.task.target_qpos = np.asarray(target_qpos)

    def step(self, action):
        assert not self._is_terminated(None)
        return np.zeros(2), 0.0, True, False, {}


def _assert_sustained(env, steps: int):
    outcomes = [env.step(np.zeros(2))[2] for _ in range(steps)]
    assert outcomes == [False] * (steps - 1) + [True]


def test_moderate_success_uses_task_specific_hold_steps():
    protocol = get_protocol("moderate")
    for env, installer, steps in (
        (FakePushT(), _install_pusht_success, 3),
        (FakeTwoRoom(), _install_tworoom_success, 3),
        (FakeReacher(), _install_reacher_success, 2),
    ):
        installer(_world(env), protocol)
        _assert_sustained(env, steps)


def test_manifest_paths_are_portable():
    from pathlib import Path

    path = Path("/machine/cache/CLEAR-LeWM/manifests/pusht/moderate.json")
    assert _portable_manifest_path(path) == "manifests/pusht/moderate.json"


def test_checkpoint_record_keeps_source_identity(tmp_path):
    directory = tmp_path / "checkpoints" / "official" / "pusht"
    directory.mkdir(parents=True)
    (directory / "weights.pt").write_bytes(b"runtime")
    (directory / "source.json").write_text(
        '{"revision": "abc", "source_weights_sha256": "def"}\n'
    )
    record = _checkpoint_record("official/pusht", tmp_path)
    assert record is not None
    assert record["source"]["revision"] == "abc"
    assert len(record["runtime_sha256"]) == 64
