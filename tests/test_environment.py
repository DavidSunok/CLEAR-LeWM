from __future__ import annotations

from clear_lewm.environment import collect_environment


def test_environment_record_covers_task_physics_and_numerics():
    record = collect_environment(task="pusht")
    assert record["task"] == "pusht"
    assert len(record["physics_fingerprint"]) == 64
    assert len(record["numerics_fingerprint"]) == 64
    assert "mujoco_runtime" in record["physics"]
    assert "pymunk" in record["physics"]
    source = record["physics"]["task_environment_source"]
    if record["packages"]["stable-worldmodel"] is not None:
        assert source["module"] == "stable_worldmodel.envs.pusht.env"
        assert len(source["sha256"]) == 64
