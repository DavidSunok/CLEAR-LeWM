from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clear_lewm.runner import (
    _fit_normalizers,
    _load_normalization_episodes,
    _normalization_row_mask,
)


class _ArrayDataset:
    """Minimal stand-in for the HDF5 dataset: column access by name."""

    def __init__(self, columns: dict[str, np.ndarray]):
        self._columns = columns

    def get_col_data(self, column: str) -> np.ndarray:
        if column not in self._columns:
            raise KeyError(column)
        return self._columns[column]


class _RecordingScaler:
    """Records the rows it was fitted on instead of computing statistics."""

    def fit(self, values: np.ndarray) -> _RecordingScaler:
        self.fitted = np.array(values, copy=True)
        return self


def _dataset() -> _ArrayDataset:
    episode = np.repeat([0, 1, 2], 4)
    action = np.arange(12, dtype=np.float64).reshape(12, 1)
    action[5, 0] = np.nan
    proprio = np.arange(24, dtype=np.float64).reshape(12, 2) * 10.0
    return _ArrayDataset(
        {
            "ep_idx": episode,
            "action": action,
            "proprio": proprio,
            "pixels": np.zeros(12),
        }
    )


def test_default_fit_uses_every_finite_row():
    process = _fit_normalizers(
        _dataset(), ["pixels", "action", "proprio"], _RecordingScaler
    )
    assert set(process) == {"action", "proprio", "goal_proprio"}
    assert process["proprio"] is process["goal_proprio"]
    assert process["action"].fitted.shape == (11, 1)
    assert not np.isnan(process["action"].fitted).any()
    assert process["proprio"].fitted.shape == (12, 2)


def test_episode_mask_restricts_the_fit_and_drops_nan_rows():
    dataset = _dataset()
    column, mask = _normalization_row_mask(dataset, np.asarray([1]))
    assert column == "ep_idx"
    assert mask.sum() == 4
    process = _fit_normalizers(dataset, ["action", "proprio"], _RecordingScaler, mask)
    # Episode 1 owns rows 4..7; row 5 has a NaN action and must be dropped.
    np.testing.assert_array_equal(process["action"].fitted[:, 0], [4.0, 6.0, 7.0])
    np.testing.assert_array_equal(
        process["proprio"].fitted, np.arange(24).reshape(12, 2)[4:8] * 10.0
    )


def test_episode_column_fallback_and_errors():
    renamed = _ArrayDataset({"episode_idx": np.repeat([7, 8], 3)})
    column, mask = _normalization_row_mask(renamed, np.asarray([8]))
    assert column == "episode_idx"
    assert mask.tolist() == [False, False, False, True, True, True]
    with pytest.raises(ValueError, match="No dataset rows"):
        _normalization_row_mask(renamed, np.asarray([99]))
    with pytest.raises(ValueError, match="no episode column"):
        _normalization_row_mask(
            _ArrayDataset({"action": np.zeros((2, 1))}), np.asarray([0])
        )


def test_load_normalization_episodes(tmp_path: Path):
    as_list = tmp_path / "list.json"
    as_list.write_text(json.dumps([3, 1, 3, 2]))
    np.testing.assert_array_equal(_load_normalization_episodes(as_list), [1, 2, 3])
    as_object = tmp_path / "object.json"
    as_object.write_text(json.dumps({"episodes": [5, 4]}))
    np.testing.assert_array_equal(_load_normalization_episodes(as_object), [4, 5])
    for bad in ([], {"other": [1]}, 5):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            _load_normalization_episodes(path)
