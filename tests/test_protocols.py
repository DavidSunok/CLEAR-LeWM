from __future__ import annotations

import pytest

from clear_lewm.protocols import (
    PRIMARY_PROTOCOLS,
    get_protocol,
    normalize_task,
    protocol_from_dict,
)


def test_protocol_contracts():
    official = get_protocol("official-compat")
    clear_id = get_protocol("clear-id")
    standard = get_protocol("clear-standard")
    assert official.sampling == "row-uniform"
    assert not official.exclude_initial_success
    assert official.reproduce_upstream_off_by_one
    assert clear_id.split == "all"
    assert clear_id.exclude_initial_success
    assert standard.sampling == "episode-balanced"
    assert standard.exclude_initial_success
    assert standard.cube_orientation_threshold_deg == 15.0
    assert standard.sustained_steps == 5


def test_primary_protocols_have_increasing_rigor():
    assert PRIMARY_PROTOCOLS == ("official", "moderate", "strict")
    official = get_protocol("official")
    moderate = get_protocol("moderate")
    strict = get_protocol("strict")
    assert official.success_mode == "upstream"
    assert moderate.sustained_steps == 3
    assert strict.sustained_steps == 5
    assert moderate.hold_steps("reacher") == 2
    assert strict.hold_steps("pusht") == 3
    assert strict.hold_steps("cube") == 5
    assert strict.tworoom_distance_threshold < moderate.tworoom_distance_threshold
    assert strict.reacher_joint_threshold_rad < moderate.reacher_joint_threshold_rad
    assert strict.pusht_position_threshold < moderate.pusht_position_threshold


def test_manifest_protocol_is_restored_without_registry_drift():
    saved = get_protocol("moderate").to_dict()
    saved["tworoom_distance_threshold"] = 11.5
    restored = protocol_from_dict(saved)
    assert restored.tworoom_distance_threshold == 11.5
    assert get_protocol("moderate").tworoom_distance_threshold == 12.0

    old_strict = get_protocol("strict").to_dict()
    old_strict.pop("pusht_sustained_steps")
    assert protocol_from_dict(old_strict).hold_steps("pusht") == 5


def test_task_aliases_and_invalid_values():
    assert normalize_task("OGBench-Cube") == "cube"
    assert normalize_task("two_room") == "tworoom"
    with pytest.raises(ValueError):
        normalize_task("unknown")
