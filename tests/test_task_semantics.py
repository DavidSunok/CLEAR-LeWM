from __future__ import annotations

import numpy as np
import pytest

from clear_lewm.tasks import (
    cube_symmetry_angle_deg,
    quaternion_angle_deg,
    wrapped_angle_error,
)


def test_cube_orientation_respects_equivalent_quarter_turns():
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])
    quarter_turn_z = np.array([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]])
    assert quaternion_angle_deg(identity, quarter_turn_z)[0] == 90.0
    assert cube_symmetry_angle_deg(identity, quarter_turn_z)[0] < 1e-6


def test_cube_orientation_retains_non_equivalent_rotation_error():
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])
    rotation_z_20 = np.array(
        [[np.cos(np.deg2rad(10.0)), 0.0, 0.0, np.sin(np.deg2rad(10.0))]]
    )
    assert cube_symmetry_angle_deg(identity, rotation_z_20)[0] == pytest.approx(20.0)


def test_wrapped_angle_error_uses_the_short_arc():
    error = wrapped_angle_error(np.array([-np.pi + 0.01]), np.array([np.pi - 0.01]))
    assert error[0] == pytest.approx(0.02)
