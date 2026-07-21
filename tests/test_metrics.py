from __future__ import annotations

import numpy as np

from clear_lewm.metrics import summarize_success, sustained_outcomes
from clear_lewm.tasks import quaternion_angle_deg


def test_sustained_success_requires_consecutive_steps():
    trace = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert sustained_outcomes(trace, 3).tolist() == [True, False, False]


def test_random_adjusted_metrics_are_paired():
    method = [True, True, False, True]
    random = [True, False, False, False]
    report = summarize_success(
        method,
        random_trace=random,
        bootstrap_samples=200,
        seed=3,
    )
    assert report["success_rate_percent"] == 75.0
    assert report["random_success_rate_percent"] == 25.0
    assert report["excess_over_random_pp"] == 50.0
    assert np.isclose(report["normalized_success_percent"], 200.0 / 3.0)


def test_quaternion_distance_handles_sign_equivalence():
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])
    negative = -identity
    quarter_turn = np.array([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]])
    assert np.isclose(quaternion_angle_deg(identity, negative)[0], 0.0)
    assert np.isclose(quaternion_angle_deg(identity, quarter_turn)[0], 90.0)
