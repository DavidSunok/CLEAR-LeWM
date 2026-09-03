from __future__ import annotations

from collections import deque
from types import MethodType
from typing import Any

import numpy as np


class ExecutedHistory:
    """Keep block-aligned observations and actions from the live rollout."""

    def __init__(self, num_envs: int, *, history_len: int, action_block: int):
        if history_len < 1:
            raise ValueError("history_len must be at least 1")
        if action_block < 1:
            raise ValueError("action_block must be at least 1")
        self.history_len = history_len
        self.action_block = action_block
        self.pixels = [deque(maxlen=history_len) for _ in range(num_envs)]
        self.actions = [deque(maxlen=max(1, history_len - 1)) for _ in range(num_envs)]
        self.pending = [[] for _ in range(num_envs)]

    def clear(self, env_i: int) -> None:
        self.pixels[env_i].clear()
        self.actions[env_i].clear()
        self.pending[env_i].clear()

    def seed_if_empty(self, env_i: int, pixels: np.ndarray) -> None:
        if not self.pixels[env_i]:
            self.pixels[env_i].append(np.asarray(pixels).copy())

    def observe(
        self,
        actions: np.ndarray,
        pixels: np.ndarray,
        active: np.ndarray,
    ) -> None:
        actions = np.asarray(actions)
        pixels = np.asarray(pixels)
        active = np.asarray(active, dtype=bool)
        for env_i in np.flatnonzero(active):
            self.pending[env_i].append(actions[env_i].copy())
            if len(self.pending[env_i]) < self.action_block:
                continue
            if len(self.pending[env_i]) != self.action_block:
                raise RuntimeError("pending action block exceeded configured size")
            self.actions[env_i].append(np.stack(self.pending[env_i]))
            self.pending[env_i].clear()
            self.pixels[env_i].append(pixels[env_i, -1].copy())


class OnlineHistoryPolicyMixin:
    """Adapt the pinned WorldModelPolicy to use executed H1-to-H3 context."""

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        self._clear_history = ExecutedHistory(
            getattr(env, "num_envs", 1),
            history_len=self.cfg.history_len,
            action_block=self.cfg.action_block,
        )
        self._clear_history_max_frames = 0

    @staticmethod
    def _slice_info(info_dict: dict, indices: list[int]) -> dict:
        import torch

        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        sliced = {}
        for key, value in info_dict.items():
            if key == "_needs_flush":
                continue
            if torch.is_tensor(value):
                sliced[key] = value[idx_tensor]
            elif isinstance(value, np.ndarray):
                sliced[key] = value[indices]
            elif isinstance(value, list):
                sliced[key] = [value[index] for index in indices]
            else:
                sliced[key] = value
        return sliced

    def _history_action_tensor(self, env_i: int):
        import torch

        blocks = list(self._clear_history.actions[env_i])
        if not blocks:
            return None
        raw = np.stack(blocks)
        shape = raw.shape
        flat = raw.reshape(-1, *shape[2:])
        if "action" in self.process:
            flat = self.process["action"].transform(flat)
        normalized = np.asarray(flat).reshape(shape)
        return torch.from_numpy(normalized.reshape(1, shape[0], -1))

    def _prepare_replan_info(self, info_dict: dict, indices: list[int]) -> dict:
        if len(indices) != 1:
            raise ValueError("online-history planning must be prepared per environment")
        env_i = indices[0]
        sliced = self._slice_info(info_dict, indices)
        current = np.asarray(sliced["pixels"])[0, -1]
        self._clear_history.seed_if_empty(env_i, current)
        history_pixels = list(self._clear_history.pixels[env_i])
        sliced["pixels"] = np.stack(history_pixels)[None, ...]
        prepared = self._prepare_info(sliced)
        history_action = self._history_action_tensor(env_i)
        if history_action is not None:
            prepared["_history_action"] = history_action
        self._clear_history_max_frames = max(
            self._clear_history_max_frames, len(history_pixels)
        )
        return prepared

    def _store_plan(self, outputs: dict, indices: list[int]) -> None:
        import torch

        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        actions = outputs["actions"]
        plan = actions[:, : self.cfg.receding_horizon]
        rest = actions[:, self.cfg.receding_horizon :]

        if self.cfg.warm_start and rest.shape[1] > 0:
            if self._next_init is None:
                self._next_init = torch.zeros(
                    len(self._action_buffer),
                    rest.shape[1],
                    rest.shape[2],
                    dtype=rest.dtype,
                )
            self._next_init[idx_tensor] = rest
        elif not self.cfg.warm_start:
            self._next_init = None

        plan = plan.reshape(len(indices), self.flatten_receding_horizon, -1)
        for row, env_i in enumerate(indices):
            self._action_buffer[env_i].extend(plan[row])

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        import torch

        if not hasattr(self, "env"):
            raise RuntimeError("Environment not set for the policy")
        n_envs = self.env.num_envs

        needs_flush = info_dict.get("_needs_flush")
        if needs_flush is not None:
            flush_mask = np.asarray(needs_flush, dtype=bool).reshape(n_envs, -1)
            flush_mask = flush_mask.any(axis=1)
            for env_i in np.flatnonzero(flush_mask):
                self._action_buffer[env_i].clear()
                self._clear_history.clear(env_i)
                if self._next_init is not None:
                    self._next_init[env_i] = 0

        terminated = info_dict.get("terminated")
        dead = (
            np.asarray(terminated, dtype=bool).reshape(n_envs, -1).any(axis=1)
            if terminated is not None
            else np.zeros(n_envs, dtype=bool)
        )
        replan_idx = [
            env_i
            for env_i in range(n_envs)
            if len(self._action_buffer[env_i]) == 0 and not dead[env_i]
        ]

        for env_i in replan_idx:
            indices = [env_i]
            prepared = self._prepare_replan_info(info_dict, indices)
            idx_tensor = torch.as_tensor(indices, dtype=torch.long)
            initial = (
                self._next_init[idx_tensor] if self._next_init is not None else None
            )
            outputs = self.solver(prepared, init_action=initial)
            self._store_plan(outputs, indices)

        action = torch.full(
            (n_envs, *self.env.single_action_space.shape),
            float("nan"),
            dtype=torch.float32,
        )
        for env_i in range(n_envs):
            if not dead[env_i]:
                action[env_i] = self._action_buffer[env_i].popleft()

        action = action.reshape(*self.env.action_space.shape).numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action

    def observe_transition(
        self,
        actions: np.ndarray,
        info_dict: dict,
        mask: np.ndarray | None = None,
    ) -> None:
        active = (
            np.ones(len(self._action_buffer), dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        reshaped_actions = np.asarray(actions).reshape(
            len(self._action_buffer), *self.env.single_action_space.shape
        )
        self._clear_history.observe(
            reshaped_actions, np.asarray(info_dict["pixels"]), active
        )


def build_online_history_policy(base_class, **kwargs):
    policy_class = type(
        "ClearOnlineHistoryPolicy",
        (OnlineHistoryPolicyMixin, base_class),
        {"__module__": __name__},
    )
    return policy_class(**kwargs)


def install_history_cost(model) -> None:
    """Prepend executed action blocks before candidate actions during rollout."""
    if getattr(model, "_clear_online_history_installed", False):
        return
    original_get_cost = model.get_cost

    def get_cost(self, info_dict: dict, action_candidates):
        import torch

        history_action = info_dict.get("_history_action")
        if history_action is None:
            return original_get_cost(info_dict, action_candidates)
        if not torch.is_tensor(history_action):
            raise TypeError("_history_action must be a torch.Tensor")
        if history_action.shape[:-2] != action_candidates.shape[:-2]:
            raise ValueError("historical and candidate action batches do not match")
        if history_action.shape[-1] != action_candidates.shape[-1]:
            raise ValueError("historical and candidate action dimensions do not match")
        rollout_actions = torch.cat((history_action, action_candidates), dim=-2)
        return original_get_cost(info_dict, rollout_actions)

    model.get_cost = MethodType(get_cost, model)
    model._clear_online_history_installed = True


def install_transition_observer(world, policy) -> None:
    """Call the adapter after every real environment transition."""
    original_step = world.envs.step

    def step(self, actions, *args, **kwargs):
        result = original_step(actions, *args, **kwargs)
        mask = kwargs.get("mask")
        if mask is None:
            mask = np.ones(world.num_envs, dtype=bool)
        policy.observe_transition(actions, result[-1], mask=mask)
        return result

    world.envs.step = MethodType(step, world.envs)
