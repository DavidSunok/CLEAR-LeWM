"""Compatibility adapter for stable-worldmodel 0.1.0's Adam planner."""

from __future__ import annotations

from typing import Any

import torch
from stable_worldmodel.solver import GradientSolver

UPSTREAM_ADAM_SOURCE = {
    "package": "stable-worldmodel",
    "version": "0.1.0",
    "class": "stable_worldmodel.solver.GradientSolver",
}


class DeviceSafeGradientSolver(GradientSolver):
    """Keep full-horizon initialization on the configured solver device.

    ``stable-worldmodel==0.1.0`` only moves the initial action tensor when it
    has to append a missing tail. Its ``prepare_init_action`` helper normally
    returns a full-horizon CPU tensor, so CUDA sampling then mixes CPU actions
    with CUDA noise. This adapter changes only that initialization boundary.
    """

    def init_action(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> None:
        if actions is None:
            actions = torch.zeros(
                (n_envs, 0, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            actions = actions.to(device=self.device, dtype=self.dtype)

        remaining = self.horizon - actions.shape[1]
        if remaining > 0:
            tail = torch.zeros(
                n_envs,
                remaining,
                self.action_dim,
                device=self.device,
                dtype=self.dtype,
            )
            actions = torch.cat((actions, tail), dim=1)

        actions = actions.unsqueeze(1).repeat_interleave(
            self.num_samples, dim=1
        )
        actions[:, 1:] += (
            torch.randn(
                actions[:, 1:].shape,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )
            * self.var_scale
        )

        if hasattr(self, "init") and self.init.shape == actions.shape:
            self.init.copy_(actions)
        else:
            if "init" in self._parameters:
                del self._parameters["init"]
            self.register_parameter("init", torch.nn.Parameter(actions))


def solver_provenance() -> dict[str, Any]:
    """Describe the narrow compatibility change applied to upstream Adam."""
    return {
        "source": UPSTREAM_ADAM_SOURCE,
        "compatibility_adapter": (
            "move full-horizon initialization to the solver device before "
            "CUDA sample noise"
        ),
    }


__all__ = [
    "DeviceSafeGradientSolver",
    "UPSTREAM_ADAM_SOURCE",
    "solver_provenance",
]
