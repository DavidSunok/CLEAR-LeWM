"""Numeric row conversion helpers shared by FAST preprocessing tools."""

from __future__ import annotations

import numpy as np


def row_array(value) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise TypeError("FAST snapshots only support numeric tensor columns")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind in ("O", "S", "U"):
        raise TypeError("FAST snapshots only support numeric tensor columns")
    return array[0] if array.shape[:1] == (1,) else array


def batch_array(name: str, value, expected_rows: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind in ("O", "S", "U"):
        raise TypeError(f"FAST column {name!r} is not numeric")
    if len(array) != expected_rows:
        raise ValueError(
            f"FAST column {name!r}: expected {expected_rows} rows, got {len(array)}"
        )
    if name == "pixels" and array.ndim == 4 and array.shape[-1] in (1, 3):
        array = np.transpose(array, (0, 3, 1, 2))
    return np.ascontiguousarray(array)


def load_contiguous_batch(
    dataset, start: int, end: int, columns: list[str]
) -> dict[str, np.ndarray]:
    expected_rows = end - start
    if hasattr(dataset, "h5_path"):
        dataset._open()
        values = {}
        cache = getattr(dataset, "_cache", {})
        for name in columns:
            source = cache[name] if name in cache else dataset.h5_file[name]
            values[name] = batch_array(name, source[start:end], expected_rows)
        return values

    indices = list(range(start, end))
    samples = (
        dataset.__getitems__(indices)
        if hasattr(dataset, "__getitems__")
        else [dataset[index] for index in indices]
    )
    return {
        name: np.stack([row_array(sample[name]) for sample in samples])
        for name in columns
    }
