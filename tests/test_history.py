from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from clear_lewm.history import (
    ExecutedHistory,
    OnlineHistoryPolicyMixin,
    build_online_history_policy,
    install_history_cost,
    install_transition_observer,
)


def _observe_block(history: ExecutedHistory, start: int) -> None:
    for step in range(start, start + 5):
        history.observe(
            actions=np.asarray([[step, -step]], dtype=np.float32),
            pixels=np.full((1, 1, 2, 2, 3), step, dtype=np.uint8),
            active=np.asarray([True]),
        )


def test_executed_history_rolls_from_h1_to_h3_on_five_step_blocks():
    history = ExecutedHistory(1, history_len=3, action_block=5)
    history.seed_if_empty(0, np.full((2, 2, 3), 99, dtype=np.uint8))

    assert len(history.pixels[0]) == 1
    assert len(history.actions[0]) == 0

    _observe_block(history, 0)
    assert len(history.pixels[0]) == 2
    assert len(history.actions[0]) == 1
    np.testing.assert_array_equal(history.pixels[0][-1], 4)
    np.testing.assert_array_equal(history.actions[0][0][:, 0], np.arange(5))

    _observe_block(history, 5)
    assert len(history.pixels[0]) == 3
    assert len(history.actions[0]) == 2
    np.testing.assert_array_equal(history.pixels[0][-1], 9)

    _observe_block(history, 10)
    assert len(history.pixels[0]) == 3
    assert len(history.actions[0]) == 2
    np.testing.assert_array_equal(history.pixels[0][0], 4)
    np.testing.assert_array_equal(history.actions[0][0][:, 0], np.arange(5, 10))


def test_policy_flattens_executed_action_blocks_for_model_context():
    torch = pytest.importorskip("torch")
    history = ExecutedHistory(1, history_len=3, action_block=5)
    history.actions[0].append(np.arange(10, dtype=np.float32).reshape(5, 2))
    history.actions[0].append(np.arange(10, 20, dtype=np.float32).reshape(5, 2))
    policy = object.__new__(OnlineHistoryPolicyMixin)
    policy._clear_history = history
    policy.process = {}

    prepared = policy._history_action_tensor(0)

    assert prepared.shape == (1, 2, 10)
    assert torch.equal(prepared[0, 0], torch.arange(10, dtype=torch.float32))


def test_history_cost_prepends_context_without_replacing_candidates():
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self):
            self.received = None

        def get_cost(self, info_dict, action_candidates):
            self.received = action_candidates
            return action_candidates.sum(dim=(-2, -1)) * 0

    model = Model()
    install_history_cost(model)
    history = torch.zeros(1, 2, 2, 10)
    candidates = torch.ones(1, 2, 5, 10)

    model.get_cost({"_history_action": history}, candidates)

    assert model.received.shape == (1, 2, 7, 10)
    assert torch.equal(model.received[..., :2, :], history)
    assert torch.equal(model.received[..., 2:, :], candidates)


def test_transition_observer_records_the_real_step_output():
    pixels = np.full((1, 1, 2, 2, 3), 7, dtype=np.uint8)

    class Pool:
        def step(self, actions, mask=None):
            return None, None, None, None, {"pixels": pixels}

    calls = []
    policy = SimpleNamespace(
        observe_transition=lambda actions, info, mask: calls.append(
            (actions.copy(), info, mask.copy())
        )
    )
    world = SimpleNamespace(envs=Pool(), num_envs=1)
    install_transition_observer(world, policy)
    actions = np.asarray([[0.25, -0.5]], dtype=np.float32)

    world.envs.step(actions)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], actions)
    np.testing.assert_array_equal(calls[0][1]["pixels"], pixels)
    np.testing.assert_array_equal(calls[0][2], [True])


def test_policy_replans_with_h1_then_executed_h2_and_h3():
    torch = pytest.importorskip("torch")

    class RecordingSolver:
        def __init__(self):
            self.calls = []

        def configure(self, *, action_space, n_envs, config):
            self.config = config

        def __call__(self, info_dict, init_action=None):
            self.calls.append(info_dict)
            return {"actions": torch.zeros(1, 5, 10)}

    class BasePolicy:
        def __init__(self, *, solver, config, process, transform):
            self.solver = solver
            self.cfg = config
            self.process = process
            self.transform = transform
            self._next_init = None

        @property
        def flatten_receding_horizon(self):
            return self.cfg.receding_horizon * self.cfg.action_block

        def set_env(self, env):
            self.env = env
            self.solver.configure(
                action_space=env.action_space,
                n_envs=env.num_envs,
                config=self.cfg,
            )
            self._action_buffer = [
                deque(maxlen=self.flatten_receding_horizon) for _ in range(env.num_envs)
            ]

        def _prepare_info(self, info_dict):
            return {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in info_dict.items()
            }

    solver = RecordingSolver()
    config = SimpleNamespace(
        history_len=3,
        action_block=5,
        horizon=5,
        receding_horizon=1,
        warm_start=False,
    )
    policy = build_online_history_policy(
        BasePolicy,
        solver=solver,
        config=config,
        process={},
        transform={},
    )
    env = SimpleNamespace(
        num_envs=1,
        action_space=SimpleNamespace(shape=(1, 2)),
        single_action_space=SimpleNamespace(shape=(2,)),
    )
    policy.set_env(env)
    info = {
        "pixels": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
        "goal": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
    }

    for block in range(3):
        for step in range(5):
            action = policy.get_action(info)
            info = {
                "pixels": np.full(
                    (1, 1, 2, 2, 3), 5 * block + step + 1, dtype=np.uint8
                ),
                "goal": info["goal"],
            }
            policy.observe_transition(action, info)

    policy.get_action(info)

    assert [call["pixels"].shape[1] for call in solver.calls] == [1, 2, 3, 3]
    assert "_history_action" not in solver.calls[0]
    assert solver.calls[1]["_history_action"].shape == (1, 1, 10)
    assert solver.calls[2]["_history_action"].shape == (1, 2, 10)
    assert solver.calls[3]["_history_action"].shape == (1, 2, 10)
