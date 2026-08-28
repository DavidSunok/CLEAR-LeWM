from __future__ import annotations

import numpy as np
import pytest

from clear_lewm.fast_conversion import batch_array
from clear_lewm.fast_profiles import FAST_TASK_PROFILES, get_fast_profile


def test_fast_profiles_cover_all_four_training_schemas():
    assert set(FAST_TASK_PROFILES) == {"pusht", "cube", "reacher", "tworoom"}
    for task, profile in FAST_TASK_PROFILES.items():
        assert profile.task == task
        assert "pixels" in profile.keys_to_load
        assert "action" in profile.keys_to_load

    cube = get_fast_profile("cube")
    assert cube.merge_mapping == {"proprio": "proprio"}


def test_fast_batch_conversion_moves_hdf5_pixels_to_channel_first():
    pixels = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    converted = batch_array("pixels", pixels, expected_rows=3)
    assert converted.shape == (3, 3, 8, 8)
    assert converted.flags.c_contiguous


def test_fast_batch_conversion_rejects_non_numeric_columns():
    with pytest.raises(TypeError, match="not numeric"):
        batch_array("task", np.asarray(["cube"]), expected_rows=1)
