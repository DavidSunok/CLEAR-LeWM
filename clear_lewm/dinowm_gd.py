"""DINO-WM optimizer profile adapted to LeWM visual latents."""

from __future__ import annotations

import time
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

DINO_WM_SOURCE = {
    "repository": "https://github.com/gaoyuezhou/dino_wm",
    "commit": "0a9492fa12044b852ae9e001cc74604b79c8bb0c",
    "planner_file": "planning/gd.py",
    "objective_file": "planning/objectives.py",
}


def solver_config() -> dict[str, Any]:
    """Return the published DINO-WM GD defaults in Hydra-compatible form."""
    return {
        "_target_": "clear_lewm.dinowm_gd.DINOWMGDPlanner",
        "model": "???",
        "n_steps": 1000,
        "batch_size": 1,
        "num_samples": 1,
        "action_noise": 0.003,
        "device": "cuda",
        "seed": "${seed}",
        "lr": 1.0,
    }


def _terminal_latent_mean_cost(
    predicted: torch.Tensor, goal: torch.Tensor
) -> torch.Tensor:
    """DINO-WM's terminal visual-latent MSE, reduced per environment/sample."""
    if goal.ndim == predicted.ndim - 1:
        goal = goal.unsqueeze(1)
    goal = goal[..., -1:, :].expand_as(predicted)
    return functional.mse_loss(
        predicted[..., -1:, :],
        goal[..., -1:, :].detach(),
        reduction="none",
    ).mean(dim=tuple(range(2, predicted.ndim)))


def install_terminal_latent_mean_criterion(model: Any) -> None:
    """Use DINO-WM's mean-reduced visual objective with a LeWM model."""

    def criterion(self, info_dict: dict, action_candidates=None):
        del action_candidates
        if "predicted_emb" in info_dict:
            predicted = info_dict["predicted_emb"]
            goal = info_dict["goal_emb"]
        else:
            predicted = info_dict["predicted_pixels_embed"]
            goal = info_dict["pixels_goal_embed"]
        return _terminal_latent_mean_cost(predicted, goal)

    model.criterion = MethodType(criterion, model)


class DINOWMGDPlanner(torch.nn.Module):
    """Apply DINO-WM's manual SGD update to a LeWM-compatible model.

    This is an optimizer-profile adapter, not a reproduction of the DINO-WM
    model. It has neither DINO-WM's encoder contract nor its proprioceptive
    objective.
    """

    def __init__(
        self,
        model: Any,
        n_steps: int = 1000,
        batch_size: int = 1,
        num_samples: int = 1,
        action_noise: float = 0.003,
        device: str | torch.device = "cuda",
        seed: int = 1234,
        lr: float = 1.0,
    ) -> None:
        super().__init__()
        if n_steps <= 0 or batch_size <= 0:
            raise ValueError("n_steps and batch_size must be positive")
        if num_samples != 1:
            raise ValueError("DINO-WM GD requires num_samples=1")
        if lr < 0 or action_noise < 0:
            raise ValueError("lr and action_noise must be non-negative")

        self.model = model
        self.n_steps = int(n_steps)
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.action_noise = float(action_noise)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.lr = float(lr)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(self.seed)
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32
        self._configured = False
        self._solve_calls = 0
        self._total_solve_time_s = 0.0
        self._finite_actions = True
        self._finite_costs = True

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        shape = tuple(action_space.shape)
        if not shape:
            raise ValueError("action_space.shape must not be empty")
        single_shape = shape[1:] if len(shape) > 1 and shape[0] == n_envs else shape
        self._single_action_dim = int(np.prod(single_shape))
        self._n_envs = int(n_envs)
        self._config = config
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def horizon(self) -> int:
        return int(self._config.horizon)

    @property
    def action_dim(self) -> int:
        return self._single_action_dim * int(self._config.action_block)

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.solve(*args, **kwargs)

    def _initialize(
        self, total_envs: int, init_action: torch.Tensor | None
    ) -> torch.Tensor:
        prefix = (
            torch.zeros(
                total_envs,
                0,
                self.action_dim,
                device=self.device,
                dtype=self._dtype,
            )
            if init_action is None
            else torch.as_tensor(init_action, device=self.device, dtype=self._dtype)
        )
        if prefix.ndim != 3 or prefix.shape[0] != total_envs:
            raise ValueError("init_action must have shape (batch, time, action_dim)")
        if prefix.shape[2] != self.action_dim or prefix.shape[1] > self.horizon:
            raise ValueError("init_action does not match configured horizon/action_dim")
        tail = torch.randn(
            total_envs,
            self.horizon - prefix.shape[1],
            self.action_dim,
            generator=self.torch_gen,
            device=self.device,
            dtype=self._dtype,
        )
        return torch.cat((prefix, tail), dim=1).unsqueeze(1)

    def _expand_info(
        self, info_dict: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        expanded = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value):
                value = value[start:end].to(self.device)
                if value.is_floating_point():
                    value = value.to(self._dtype)
                expanded[key] = value.unsqueeze(1)
            elif isinstance(value, np.ndarray):
                expanded[key] = value[start:end, None, ...]
            elif isinstance(value, (list, tuple)):
                expanded[key] = value[start:end]
            else:
                expanded[key] = value
        return expanded

    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        if not self._configured:
            raise RuntimeError("configure() must be called before solve()")

        started = time.perf_counter()
        if init_action is not None:
            total_envs = int(init_action.shape[0])
        else:
            try:
                total_envs = len(info_dict["pixels"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "info_dict['pixels'] must expose the current batch size"
                ) from exc
        initial = self._initialize(total_envs, init_action)
        selected = []
        histories = []

        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            actions = initial[start:end].clone().detach().requires_grad_(True)
            optimizer = torch.optim.SGD([actions], lr=self.lr)
            batch_info = self._expand_info(info_dict, start, end)
            history = []

            for _ in range(self.n_steps):
                optimizer.zero_grad()
                costs = self.model.get_cost(dict(batch_info), actions)
                expected = (end - start, 1)
                if not torch.is_tensor(costs) or tuple(costs.shape) != expected:
                    raise ValueError(f"model cost shape must be {expected}")
                total_loss = costs.mean() * (end - start)
                total_loss.backward()
                with torch.no_grad():
                    actions_new = (
                        actions - optimizer.param_groups[0]["lr"] * actions.grad
                    )
                    if self.action_noise:
                        noise = torch.randn(
                            actions.shape,
                            generator=self.torch_gen,
                            device=self.device,
                            dtype=actions.dtype,
                        )
                        actions_new.add_(noise, alpha=self.action_noise)
                    actions.copy_(actions_new)
                history.append(float(total_loss.detach().cpu()))

            selected.append(actions.detach()[:, 0].cpu())
            histories.append(history)

        actions_out = torch.cat(selected, dim=0)
        solve_time_s = time.perf_counter() - started
        self._solve_calls += 1
        self._total_solve_time_s += solve_time_s
        self._finite_actions = self._finite_actions and bool(
            torch.isfinite(actions_out).all()
        )
        self._finite_costs = self._finite_costs and bool(
            np.isfinite(np.asarray(histories, dtype=np.float64)).all()
        )
        return {
            "actions": actions_out,
            "cost": histories,
            "solve_time_s": solve_time_s,
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return compact finite-value and timing telemetry for provenance."""
        return {
            "solve_calls": self._solve_calls,
            "total_solve_time_s": self._total_solve_time_s,
            "finite_actions": self._finite_actions,
            "finite_costs": self._finite_costs,
        }


__all__ = [
    "DINO_WM_SOURCE",
    "DINOWMGDPlanner",
    "install_terminal_latent_mean_criterion",
    "solver_config",
]
