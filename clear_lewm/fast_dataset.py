"""Auditable memory-mapped reader for preprocessed LeWM datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    from stable_worldmodel.data.dataset import Dataset
except ImportError as exc:  # pragma: no cover - exercised without the optional extra
    raise ImportError("FastMemmapDataset requires `pip install -e '.[lewm]'`.") from exc


class FastMemmapDataset(Dataset):
    """Read a verified row-major tensor snapshot without image decoding."""

    def __init__(
        self,
        directory: str | Path,
        num_steps: int = 1,
        frameskip: int = 1,
        transform=None,
    ) -> None:
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "meta.json").read_text())
        if self.meta.get("schema_version") != "clear-fast-memmap-v1":
            raise ValueError("Unsupported or missing FAST memmap schema version")
        if not self.meta.get("complete", False):
            raise ValueError("FAST memmap conversion is incomplete")

        self._arrays: dict[str, np.memmap] = {}
        for name, shape in self.meta["shapes"].items():
            dtype = np.dtype(self.meta["dtypes"][name])
            path = self.directory / f"{name}.bin"
            expected_bytes = int(np.prod(shape)) * dtype.itemsize
            if path.stat().st_size != expected_bytes:
                actual_bytes = path.stat().st_size
                raise ValueError(
                    f"{name}: expected {expected_bytes} bytes, got {actual_bytes}"
                )
            self._arrays[name] = np.memmap(
                path, dtype=dtype, mode="r", shape=tuple(shape)
            )

        lengths = np.load(self.directory / "episode_lengths.npy")
        offsets = np.load(self.directory / "episode_offsets.npy")
        if int(lengths.sum()) != int(self.meta["rows"]):
            raise ValueError("Episode lengths do not sum to the declared row count")
        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    @property
    def column_names(self) -> list[str]:
        return list(self._arrays)

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        row_start = int(self.offsets[ep_idx] + start)
        row_end = row_start + (end - start)
        steps = {}
        for name, array in self._arrays.items():
            values = (
                array[row_start:row_end]
                if name == "action"
                else array[row_start : row_end : self.frameskip]
            )
            steps[name] = torch.from_numpy(np.asarray(values).copy())
        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        return np.asarray(self._arrays[col])

    def get_dim(self, col: str) -> int:
        shape = self.meta["shapes"][col]
        return int(shape[-1]) if len(shape) > 1 else 1

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {
            name: torch.from_numpy(np.asarray(array[row_idx]).copy())
            for name, array in self._arrays.items()
        }


FastPushTDataset = FastMemmapDataset
