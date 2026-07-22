from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from clear_lewm.cli import build_parser
from clear_lewm.protocols import get_protocol
from clear_lewm.runner import (
    _audit_checkpoint_state,
    _checkpoint_record,
    _install_batched_lewm_criterion,
    _install_pusht_success,
    _install_reacher_success,
    _install_tworoom_success,
    _load_paired_random_trace,
    _portable_manifest_path,
)
from clear_lewm.runtime import audit_hydra_targets, configure_import_paths


def _world(env):
    wrapped = SimpleNamespace(unwrapped=env)
    return SimpleNamespace(envs=SimpleNamespace(envs=[wrapped]))


@pytest.fixture
def isolated_legacy_module():
    original = sys.modules.pop("module", None)
    try:
        yield
    finally:
        sys.modules.pop("module", None)
        if original is not None:
            sys.modules["module"] = original


class FakePushT:
    def __init__(self):
        self.goal_state = np.zeros(7)
        self.state = np.zeros(7)

    def _set_goal_state(self, goal_state):
        self.goal_state = np.asarray(goal_state)

    def step(self, action):
        return {"state": self.state.copy()}, 0.0, True, False, {}


class FakeTwoRoom:
    def __init__(self):
        self.agent_position = np.zeros(2)
        self.target_position = np.zeros(2)

    def _set_goal_state(self, goal_state):
        self.target_position = np.asarray(goal_state)

    def step(self, action):
        return np.zeros(2), 0.0, True, False, {}


class FakeReacher:
    def __init__(self):
        physics = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(2)))
        task = SimpleNamespace(target_qpos=np.zeros(2))
        self.env = SimpleNamespace(physics=physics, task=task)

    def _is_terminated(self, step):
        return True

    def set_target_qpos(self, target_qpos):
        self.env.task.target_qpos = np.asarray(target_qpos)

    def step(self, action):
        assert not self._is_terminated(None)
        return np.zeros(2), 0.0, True, False, {}


def _assert_sustained(env, steps: int):
    outcomes = [env.step(np.zeros(2))[2] for _ in range(steps)]
    assert outcomes == [False] * (steps - 1) + [True]


def test_moderate_success_uses_task_specific_hold_steps():
    protocol = get_protocol("moderate")
    for env, installer, steps in (
        (FakePushT(), _install_pusht_success, 3),
        (FakeTwoRoom(), _install_tworoom_success, 3),
        (FakeReacher(), _install_reacher_success, 1),
    ):
        installer(_world(env), protocol)
        _assert_sustained(env, steps)


def test_pusht_task_semantics_ignore_the_final_pusher_position():
    env = FakePushT()
    env.goal_state = np.array([100.0, 100.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    env.state = np.array([0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    _install_pusht_success(_world(env), get_protocol("moderate"))
    _assert_sustained(env, 3)


def test_reacher_uses_shortest_periodic_joint_error():
    env = FakeReacher()
    env.env.physics.data.qpos = np.array([-np.pi + 0.01, 0.0])
    env.env.task.target_qpos = np.array([np.pi - 0.01, 0.0])
    _install_reacher_success(_world(env), get_protocol("strict"))
    assert env.step(np.zeros(2))[2]


def test_manifest_paths_are_portable():
    from pathlib import Path

    path = Path("/machine/cache/CLEAR-LeWM/manifests/pusht/moderate.json")
    assert _portable_manifest_path(path) == "manifests/pusht/moderate.json"


def test_checkpoint_record_keeps_source_identity(tmp_path):
    directory = tmp_path / "checkpoints" / "official" / "pusht"
    directory.mkdir(parents=True)
    (directory / "weights.pt").write_bytes(b"runtime")
    (directory / "source.json").write_text(
        '{"revision": "abc", "source_weights_sha256": "def"}\n'
    )
    record = _checkpoint_record("official/pusht", tmp_path)
    assert record is not None
    assert record["source"]["revision"] == "abc"
    assert len(record["runtime_sha256"]) == 64


def test_batched_lewm_criterion_adds_the_missing_sample_axis():
    torch = pytest.importorskip("torch")
    model = SimpleNamespace()
    _install_batched_lewm_criterion(model)
    predicted = torch.zeros(4, 3, 2, 5)
    goal = torch.ones(4, 1, 5)
    costs = model.criterion({"predicted_emb": predicted, "goal_emb": goal})
    assert costs.shape == (4, 3)
    assert torch.equal(costs, torch.full((4, 3), 5.0))


def test_strict_checkpoint_audit_rejects_missing_keys(tmp_path):
    torch = pytest.importorskip("torch")
    directory = tmp_path / "checkpoints" / "run"
    directory.mkdir(parents=True)
    torch.save({"weight": torch.ones(1, 1)}, directory / "weights.pt")
    model = torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="Strict checkpoint audit failed"):
        _audit_checkpoint_state(model, "run", tmp_path, strict=True)


def test_custom_runtime_has_priority_and_target_provenance(
    tmp_path, monkeypatch, isolated_legacy_module
):
    runtime = tmp_path / "runtime"
    upstream = tmp_path / "upstream"
    runtime.mkdir()
    upstream.mkdir()
    (runtime / "module.py").write_text("class InverseTransitionActor:\n    pass\n")
    (upstream / "module.py").write_text("class OldActor:\n    pass\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"_target_": "module.InverseTransitionActor"}))

    monkeypatch.setattr(sys, "path", list(sys.path))
    paths = configure_import_paths(upstream, runtime)
    audit = audit_hydra_targets(config, upstream, runtime)

    assert paths["custom_runtime"] is True
    assert sys.path[0] == str(runtime)
    assert sys.path[-1] == str(upstream)
    assert audit["custom_runtime_verified"] is True
    assert audit["targets"][0]["source"]["scope"] == "runtime"
    assert len(audit["targets"][0]["source"]["sha256"]) == 64


def test_target_audit_rejects_cached_module_outside_runtime(
    tmp_path, monkeypatch, isolated_legacy_module
):
    runtime = tmp_path / "runtime"
    upstream = tmp_path / "upstream"
    runtime.mkdir()
    upstream.mkdir()
    for directory in (runtime, upstream):
        (directory / "module.py").write_text(
            "class InverseTransitionActor:\n    pass\n"
        )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"_target_": "module.InverseTransitionActor"}))

    monkeypatch.setattr(sys, "path", [str(upstream), *sys.path])
    __import__("module")
    configure_import_paths(upstream, runtime)
    with pytest.raises(RuntimeError, match="resolved outside the requested runtime"):
        audit_hydra_targets(config, upstream, runtime)


def test_paired_random_result_requires_the_same_manifest(tmp_path):
    result = {
        "schema_version": "clear-lewm-result-v1",
        "manifest_sha256": "wrong",
        "task": "pusht",
        "protocol": {"name": "moderate"},
        "policy_seed": 42,
        "checkpoint": None,
        "episode_successes": [False, True],
    }
    path = tmp_path / "random.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="manifest_sha256"):
        _load_paired_random_trace(
            path,
            manifest_sha256="expected",
            task="pusht",
            protocol_name="moderate",
            policy_seed=42,
        )


def test_paired_random_result_accepts_an_identical_run_identity(tmp_path):
    result = {
        "schema_version": "clear-lewm-result-v1",
        "manifest_sha256": "same",
        "task": "cube",
        "protocol": {"name": "strict"},
        "policy_seed": 7,
        "checkpoint": None,
        "episode_successes": [False, True],
    }
    path = tmp_path / "random.json"
    path.write_text(json.dumps(result))
    trace = _load_paired_random_trace(
        path,
        manifest_sha256="same",
        task="cube",
        protocol_name="strict",
        policy_seed=7,
    )
    assert trace == [False, True]


def test_actor_warmstart_cli_is_explicit_and_defaults_to_auto():
    parser = build_parser()
    common = ["evaluate", "--manifest", "manifest.json", "--output", "out.json"]
    assert parser.parse_args(common).actor_warmstart == "auto"
    assert (
        parser.parse_args([*common, "--actor-warmstart", "off"]).actor_warmstart
        == "off"
    )


def test_direct_cli_records_an_explicit_target_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--manifest",
            "manifest.json",
            "--output",
            "out.json",
            "--inference-mode",
            "direct",
            "--direct-target-mode",
            "goal",
            "--actor-warmstart",
            "on",
        ]
    )
    assert args.inference_mode == "direct"
    assert args.direct_target_mode == "goal"
    assert args.actor_warmstart == "on"
