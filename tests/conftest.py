from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def cube_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "cube.h5"
    episodes = 20
    length = 32
    total = episodes * length
    ep_idx = np.repeat(np.arange(100, 100 + episodes), length)
    step_idx = np.tile(np.arange(length), episodes)
    position = np.zeros((total, 3), dtype=np.float64)
    quaternion = np.zeros((total, 4), dtype=np.float64)
    quaternion[:, 0] = 1.0

    for episode in range(episodes):
        rows = slice(episode * length, (episode + 1) * length)
        if episode % 2 == 0:
            position[rows, 0] = episode
        else:
            position[rows, 0] = episode + np.arange(length) * 0.05

    with h5py.File(path, "w") as dataset:
        dataset["ep_idx"] = ep_idx
        dataset["step_idx"] = step_idx
        dataset["privileged_block_0_pos"] = position
        dataset["privileged_block_0_quat"] = quaternion
        dataset["action"] = np.zeros((total, 5), dtype=np.float32)
    return path
