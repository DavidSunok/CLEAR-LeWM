from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_worldmodel")
FastMemmapDataset = pytest.importorskip("clear_lewm.fast_dataset").FastMemmapDataset


def _write_raw(path, values):
    array = np.memmap(path, dtype=values.dtype, mode="w+", shape=values.shape)
    array[:] = values
    array.flush()


def test_fast_memmap_preserves_transform_and_action_chunking(tmp_path):
    rows = 6
    pixels = np.arange(rows * 3 * 4 * 4, dtype=np.uint8).reshape(rows, 3, 4, 4)
    actions = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
    states = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    values = {"pixels": pixels, "action": actions, "state": states}
    for name, value in values.items():
        _write_raw(tmp_path / f"{name}.bin", value)
    np.save(tmp_path / "episode_lengths.npy", np.array([rows], dtype=np.int64))
    np.save(tmp_path / "episode_offsets.npy", np.array([0], dtype=np.int64))
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": "clear-fast-memmap-v1",
                "complete": True,
                "rows": rows,
                "shapes": {name: list(value.shape) for name, value in values.items()},
                "dtypes": {name: str(value.dtype) for name, value in values.items()},
            }
        )
    )

    calls = []

    def transform(sample):
        calls.append(True)
        return {**sample, "action": sample["action"] + 10}

    dataset = FastMemmapDataset(tmp_path, num_steps=2, frameskip=2, transform=transform)
    sample = dataset[0]
    assert calls == [True]
    torch.testing.assert_close(
        sample["action"], torch.from_numpy(actions[:4]).reshape(2, 4) + 10
    )
    torch.testing.assert_close(sample["pixels"], torch.from_numpy(pixels[[0, 2]]))
