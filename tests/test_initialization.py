from __future__ import annotations

import numpy as np
import pytest

from clear_lewm.initialization import prepare_initialization


class FakeDataset:
    column_names = ["state", "qpos", "qvel", "pixels"]

    def __init__(self, chunk):
        self.chunk = chunk

    def load_chunk(self, episodes_idx, start, end):
        return [self.chunk]


def test_pusht_rest_start_preserves_configuration_and_goal():
    state = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 0.5, 4.0, -3.0],
            [11.0, 21.0, 31.0, 41.0, 0.6, 5.0, -2.0],
        ]
    )
    dataset = FakeDataset({"state": state})
    prepared, audit = prepare_initialization(dataset, "pusht", "rest-start")

    output = prepared.load_chunk([0], [5], [7])[0]["state"]

    np.testing.assert_array_equal(output[0, :5], state[0, :5])
    np.testing.assert_array_equal(output[0, -2:], np.zeros(2))
    np.testing.assert_array_equal(output[-1], state[-1])
    np.testing.assert_array_equal(dataset.chunk["state"], state)
    assert audit == {
        "mode": "rest-start",
        "velocity_field": "state[-2:]",
        "zeroed_pairs": 1,
    }


@pytest.mark.parametrize("task", ["cube", "reacher"])
def test_qvel_rest_start_zeros_only_initial_velocity(task):
    qpos = np.array([[0.2, -0.3], [0.4, -0.1]])
    qvel = np.array([[2.0, -1.0], [3.0, 4.0]])
    dataset = FakeDataset({"qpos": qpos, "qvel": qvel})
    prepared, audit = prepare_initialization(dataset, task, "rest-start")

    output = prepared.load_chunk([0], [5], [7])[0]

    np.testing.assert_array_equal(output["qpos"], qpos)
    np.testing.assert_array_equal(output["qvel"][0], np.zeros(2))
    np.testing.assert_array_equal(output["qvel"][-1], qvel[-1])
    np.testing.assert_array_equal(dataset.chunk["qvel"], qvel)
    assert audit["velocity_field"] == "qvel"
    assert audit["zeroed_pairs"] == 1


def test_recorded_state_and_tworoom_are_not_rewritten():
    dataset = FakeDataset({"state": np.ones((2, 7))})

    recorded, recorded_audit = prepare_initialization(
        dataset, "pusht", "recorded-state"
    )
    tworoom, tworoom_audit = prepare_initialization(dataset, "tworoom", "rest-start")

    assert recorded is dataset
    assert recorded_audit["zeroed_pairs"] == 0
    assert tworoom is dataset
    assert tworoom_audit == {
        "mode": "rest-start",
        "velocity_field": None,
        "zeroed_pairs": 0,
    }
