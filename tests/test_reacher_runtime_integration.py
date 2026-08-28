from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from clear_lewm.protocols import get_protocol
from clear_lewm.runner import _install_reacher_success

os.environ.setdefault("MUJOCO_GL", "egl")


def _world(env):
    wrapped = SimpleNamespace(unwrapped=env)
    return SimpleNamespace(envs=SimpleNamespace(envs=[wrapped]))


def _reacher_class():
    pytest.importorskip("dm_control")
    module = pytest.importorskip("stable_worldmodel.envs.dmcontrol.reacher")
    return module.ReacherDMControlWrapper


def _place_at_target(env):
    env.env.physics.data.qvel[:] = 0.0
    env.env.physics.forward()
    target = env.env.physics.data.qpos.copy()
    env.set_target_qpos(target)
    return target


def test_reacher_action_repeat_does_not_autoreset():
    env = _reacher_class()(task="qpos_match", seed=0)
    _install_reacher_success(_world(env), get_protocol("moderate"))
    env.reset(seed=0)
    target = _place_at_target(env)
    step_count = env.env._step_count

    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float64))

    assert env.action_repeat == 2
    assert env.env._step_count == step_count + env.action_repeat
    assert np.max(np.abs(env.env.physics.data.qpos - target)) < 0.05
    assert terminated
    assert not truncated


def test_reacher_gate_survives_recompile():
    env = _reacher_class()(task="qpos_match", seed=0)
    _install_reacher_success(_world(env), get_protocol("moderate"))
    original_task = env.env.task

    env.compile_model(seed=1, environment_kwargs={})

    assert env.env.task is not original_task
    assert env.env.task._clear_lewm_termination_disabled
    env.reset(seed=1)
    target = _place_at_target(env)
    _, _, terminated, _, _ = env.step(np.zeros(2, dtype=np.float64))
    assert np.max(np.abs(env.env.physics.data.qpos - target)) < 0.05
    assert terminated
