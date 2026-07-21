from __future__ import annotations

import pytest

from clear_lewm import environment
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
    assert len(record["execution_fingerprint"]) == 64
    assert set(record["execution"]["sources"]) == {
        "cem",
        "checkpoint_loader",
        "policy",
        "world",
    }


def test_official_runtime_audit_rejects_source_drift(monkeypatch):
    expected = environment.OFFICIAL_RUNTIME_HASHES["0.1.0"]
    monkeypatch.setattr(environment, "_version", lambda name: "0.1.0")
    monkeypatch.setattr(
        environment,
        "_module_source",
        lambda module: {
            "module": module,
            "file": module.rsplit(".", 1)[-1] + ".py",
            "sha256": (
                "modified"
                if module == "stable_worldmodel.world.world"
                else expected[
                    next(
                        name
                        for name, target in (
                            environment.EVALUATION_RUNTIME_MODULES.items()
                        )
                        if target == module
                    )
                ]
            ),
        },
    )

    audit = environment.stable_worldmodel_source_audit()
    assert audit["status"] == "mismatch"
    assert audit["mismatches"] == ["world"]
    with pytest.raises(RuntimeError, match="differences=world"):
        environment.require_official_stable_worldmodel()
