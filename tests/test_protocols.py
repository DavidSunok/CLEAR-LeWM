from __future__ import annotations

import pytest

from clear_lewm.protocols import get_protocol, normalize_task


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


def test_task_aliases_and_invalid_values():
    assert normalize_task("OGBench-Cube") == "cube"
    assert normalize_task("two_room") == "tworoom"
    with pytest.raises(ValueError):
        normalize_task("unknown")
