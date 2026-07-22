from __future__ import annotations

import numpy as np

from clear_lewm.topology import (
    TwoRoomGeometry,
    check_route_segment,
    point_is_clear,
    resolve_motion,
)


def geometry(door_half_extent: float = 14.0) -> TwoRoomGeometry:
    return TwoRoomGeometry(
        image_size=224.0,
        border_size=14.0,
        wall_center=112.0,
        wall_axis=1,
        wall_thickness=10.0,
        agent_radius=7.0,
        doors=((49.0, door_half_extent),),
    )


def test_solid_wall_crossing_is_blocked():
    result = resolve_motion(
        geometry(), np.array([90.0, 100.0]), np.array([130.0, 100.0])
    )
    assert result.blocked
    assert result.position[0] < 112.0
    assert check_route_segment(
        geometry(), np.array([90.0, 100.0]), np.asarray(result.position)
    ).valid


def test_wide_door_crossing_is_allowed():
    result = resolve_motion(geometry(), np.array([90.0, 49.0]), np.array([130.0, 49.0]))
    assert not result.blocked
    assert result.position[0] == 130.0
    assert check_route_segment(
        geometry(), np.array([90.0, 49.0]), np.asarray(result.position)
    ).valid


def test_visible_door_edge_without_radius_clearance_is_blocked():
    start = np.array([90.0, 60.0])
    desired = np.array([130.0, 60.0])
    assert not check_route_segment(geometry(), start, desired).valid
    result = resolve_motion(geometry(), start, desired)
    assert result.blocked
    assert result.position[0] < 112.0


def test_narrow_door_cannot_pass_the_agent_disk():
    result = resolve_motion(
        geometry(door_half_extent=6.0),
        np.array([90.0, 49.0]),
        np.array([130.0, 49.0]),
    )
    assert result.blocked
    assert result.position[0] < 112.0


def test_diagonal_segment_cannot_tunnel_through_corner():
    start = np.array([90.0, 70.0])
    desired = np.array([130.0, 55.0])
    result = resolve_motion(geometry(), start, desired)
    assert result.blocked
    assert check_route_segment(geometry(), start, np.asarray(result.position)).valid


def test_start_overlapping_a_doorframe_is_not_route_valid():
    start = np.array([107.9, 61.0])
    assert not point_is_clear(geometry(), start)
    check = check_route_segment(geometry(), start, start + np.array([0.1, 0.0]))
    assert not check.valid
    assert not check.start_clear
