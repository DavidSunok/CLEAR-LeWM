from __future__ import annotations

from typing import Any

import numpy as np

RECORDED_STATE = "recorded-state"
REST_START = "rest-start"
INITIALIZATION_MODES = (RECORDED_STATE, REST_START)

_VELOCITY_FIELDS = {
    "pusht": "state[-2:]",
    "cube": "qvel",
    "reacher": "qvel",
    "tworoom": None,
}


def _copy_array(value: Any):
    if hasattr(value, "clone"):
        return value.clone()
    return np.array(value, copy=True)


class RestStartDataset:
    """Present recorded configurations with zero initial dynamic velocity."""

    def __init__(self, dataset, task: str, audit: dict):
        self._dataset = dataset
        self._task = task
        self._audit = audit

    def __getattr__(self, name: str):
        return getattr(self._dataset, name)

    def load_chunk(self, episodes_idx, start, end) -> list[dict]:
        chunks = self._dataset.load_chunk(episodes_idx, start, end)
        if self._task == "tworoom":
            return chunks

        field = "state" if self._task == "pusht" else "qvel"
        transformed = []
        for chunk in chunks:
            if field not in chunk:
                raise KeyError(
                    f"Rest-start requires dataset field {field!r} for {self._task}"
                )
            values = _copy_array(chunk[field])
            if values.shape[0] == 0:
                raise ValueError("Rest-start received an empty evaluation chunk")
            if self._task == "pusht":
                if values.shape[-1] < 7:
                    raise ValueError(
                        "PushT rest-start requires the seven-dimensional state "
                        "containing agent velocity"
                    )
                values[0, -2:] = 0
            else:
                values[0] = 0
            copied = dict(chunk)
            copied[field] = values
            transformed.append(copied)

        self._audit["zeroed_pairs"] += len(transformed)
        return transformed


def prepare_initialization(dataset, task: str, mode: str):
    if mode not in INITIALIZATION_MODES:
        choices = ", ".join(INITIALIZATION_MODES)
        raise ValueError(
            f"Unknown initialization mode {mode!r}. Choose one of: {choices}"
        )

    audit = {
        "mode": mode,
        "velocity_field": _VELOCITY_FIELDS[task],
        "zeroed_pairs": 0,
    }
    if mode == RECORDED_STATE or _VELOCITY_FIELDS[task] is None:
        return dataset, audit
    return RestStartDataset(dataset, task, audit), audit
