from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from clear_lewm.history import build_online_history_policy
from clear_lewm.initialization import prepare_initialization


def test_reacher_rest_start_reaches_the_pinned_world_state_extractor():
    torch = pytest.importorskip("torch")
    world_module = pytest.importorskip("stable_worldmodel.world.world")

    class Dataset:
        column_names = ["pixels", "qpos", "qvel"]

        def load_chunk(self, episodes_idx, start, end):
            return [
                {
                    "pixels": torch.zeros(3, 3, 4, 4),
                    "qpos": torch.tensor([[0.2, -0.3], [0.3, -0.2], [0.4, -0.1]]),
                    "qvel": torch.tensor([[2.0, -1.0], [3.0, 1.0], [4.0, 2.0]]),
                }
            ]

    prepared, audit = prepare_initialization(Dataset(), "reacher", "rest-start")
    init_state, goal_state, _ = world_module._extract_init_goal(
        prepared,
        episodes_idx=[0],
        start_steps=[5],
        goal_offset=2,
    )

    np.testing.assert_allclose(init_state["qpos"], [[0.2, -0.3]])
    np.testing.assert_array_equal(init_state["qvel"], [[0.0, 0.0]])
    np.testing.assert_allclose(goal_state["goal_qvel"], [[4.0, 2.0]])
    assert audit["zeroed_pairs"] == 1


def test_pinned_world_model_policy_replans_with_executed_h3():
    gym_spaces = pytest.importorskip("gymnasium.spaces")
    torch = pytest.importorskip("torch")
    policy_module = pytest.importorskip("stable_worldmodel.policy")

    class RecordingSolver:
        def __init__(self):
            self.calls = []

        def configure(self, *, action_space, n_envs, config):
            self._n_envs = n_envs
            self.config = config

        @property
        def n_envs(self):
            return self._n_envs

        @property
        def horizon(self):
            return self.config.horizon

        @property
        def action_dim(self):
            return 10

        def solve(self, info_dict, init_action=None):
            self.calls.append(info_dict)
            return {"actions": torch.arange(50, dtype=torch.float32).reshape(1, 5, 10)}

        def __call__(self, info_dict, init_action=None):
            return self.solve(info_dict, init_action)

    solver = RecordingSolver()
    policy = build_online_history_policy(
        policy_module.WorldModelPolicy,
        solver=solver,
        config=policy_module.PlanConfig(
            horizon=5,
            receding_horizon=5,
            history_len=3,
            action_block=5,
        ),
        process={},
        transform={},
    )
    env = SimpleNamespace(
        num_envs=1,
        action_space=gym_spaces.Box(low=-1, high=1, shape=(1, 2), dtype=np.float32),
        single_action_space=gym_spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        ),
    )
    policy.set_env(env)
    info = {
        "pixels": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
        "goal": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
    }

    for step in range(25):
        action = policy.get_action(info)
        info = {
            "pixels": np.full((1, 1, 2, 2, 3), step + 1, dtype=np.uint8),
            "goal": info["goal"],
        }
        policy.observe_transition(action, info)
    policy.get_action(info)

    assert [call["pixels"].shape[1] for call in solver.calls] == [1, 3]
    np.testing.assert_array_equal(
        solver.calls[1]["pixels"][:, :, 0, 0, 0], [[15, 20, 25]]
    )
    assert solver.calls[1]["_history_action"].shape == (1, 2, 10)
    assert torch.equal(
        solver.calls[1]["_history_action"],
        torch.arange(30, 50, dtype=torch.float32).reshape(1, 2, 10),
    )
