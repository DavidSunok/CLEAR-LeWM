from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clear_lewm.cli import build_parser
from clear_lewm.protocols import get_protocol
from clear_lewm.runner import (
    _audit_checkpoint_state,
    _checkpoint_record,
    _compose_config,
    _install_batched_lewm_criterion,
    _install_pusht_success,
    _install_reacher_success,
    _install_tworoom_success,
    _load_paired_random_trace,
    _portable_manifest_path,
    evaluate_manifest,
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


class FakeGeomPositions:
    def __init__(self, physics):
        self.physics = physics

    def __getitem__(self, key):
        name, axes = key
        assert name == "finger"
        q0, q1 = self.physics.data.qpos
        position = np.array(
            [np.cos(q0) + np.cos(q0 + q1), np.sin(q0) + np.sin(q0 + q1), 0.0]
        )
        return position[axes]


class FakePhysics:
    def __init__(self):
        self.data = SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2))
        self.named = SimpleNamespace(
            data=SimpleNamespace(geom_xpos=FakeGeomPositions(self))
        )

    def forward(self):
        return None


class FakeReacherTask:
    def __init__(self):
        self.target_qpos = np.zeros(2)

    def get_termination(self, physics):
        return 0.0


class FakeReacher:
    def __init__(self):
        self.compile_model()

    def compile_model(self, *args, **kwargs):
        self.env = SimpleNamespace(physics=FakePhysics(), task=FakeReacherTask())

    def reset(self, *args, **kwargs):
        self.compile_model()
        return np.zeros(2), {}

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
        (FakePushT(), _install_pusht_success, 1),
        (FakeTwoRoom(), _install_tworoom_success, 1),
        (FakeReacher(), _install_reacher_success, 1),
    ):
        installer(_world(env), protocol)
        _assert_sustained(env, steps)


def test_pusht_moderate_keeps_the_official_pusher_and_block_state():
    env = FakePushT()
    env.goal_state = np.array([100.0, 100.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    env.state = np.array([0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    _install_pusht_success(_world(env), get_protocol("moderate"))
    assert not env.step(np.zeros(2))[2]
    env.state[:2] = env.goal_state[:2]
    assert env.step(np.zeros(2))[2]


def test_reacher_uses_shortest_periodic_joint_error():
    env = FakeReacher()
    env.env.physics.data.qpos = np.array([-np.pi + 0.01, 0.0])
    env.env.task.target_qpos = np.array([np.pi - 0.01, 0.0])
    protocol = replace(get_protocol("moderate"), reacher_angle_mode="all-periodic")
    _install_reacher_success(_world(env), protocol)
    assert env.step(np.zeros(2))[2]


def test_reacher_moderate_wraps_shoulder_but_not_bounded_wrist():
    env = FakeReacher()
    env.env.physics.data.qpos = np.array([-np.pi + 0.01, -2.7])
    env.env.task.target_qpos = np.array([np.pi - 0.01, 2.7])
    _install_reacher_success(_world(env), get_protocol("moderate"))
    assert not env.step(np.zeros(2))[2]
    env.env.physics.data.qpos[1] = env.env.task.target_qpos[1]
    assert env.step(np.zeros(2))[2]


def test_reacher_strict_scores_only_the_fingertip_endpoint():
    env = FakeReacher()
    _install_reacher_success(_world(env), get_protocol("strict"))
    env.set_target_qpos(np.array([0.0, 0.0]))
    env.env.physics.data.qpos[:] = np.array([0.1, 0.0])
    assert not env.step(np.zeros(2))[2]
    env.env.physics.data.qpos[:] = np.array([0.0, 0.0])
    assert not env.step(np.zeros(2))[2]
    assert env.step(np.zeros(2))[2]


def test_reacher_task_termination_stays_disabled_after_rebuilds():
    env = FakeReacher()
    _install_reacher_success(_world(env), get_protocol("moderate"))

    for rebuild in (None, env.reset, env.compile_model):
        if rebuild is not None:
            rebuild()
        assert env.env.task._clear_lewm_termination_disabled
        assert env.env.task.get_termination(env.env.physics) is None


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


def test_planner_cli_defaults_to_cem_and_accepts_alternatives():
    parser = build_parser()
    common = ["evaluate", "--manifest", "manifest.json", "--output", "out.json"]
    assert parser.parse_args(common).planner == "cem"
    assert parser.parse_args([*common, "--planner", "adam"]).planner == "adam"
    assert parser.parse_args([*common, "--planner", "dinowm-gd"]).planner == "dinowm-gd"


def test_compose_config_selects_upstream_planner():
    pytest.importorskip("hydra")
    pytest.importorskip("torch")
    upstream = Path(__file__).resolve().parents[1] / "third_party" / "le-wm"
    cem = _compose_config("pusht", upstream, planner="cem")
    adam = _compose_config("pusht", upstream, planner="adam")
    dinowm_gd = _compose_config("pusht", upstream, planner="dinowm-gd")
    assert cem.solver._target_ == "stable_worldmodel.solver.CEMSolver"
    assert adam.solver._target_ == "stable_worldmodel.solver.GradientSolver"
    assert dinowm_gd.solver._target_ == "clear_lewm.dinowm_gd.DINOWMGDPlanner"
    assert dinowm_gd.solver.n_steps == 1000
    assert dinowm_gd.solver.lr == 1.0
    assert dinowm_gd.solver.action_noise == 0.003


def test_dinowm_objective_means_over_terminal_latent_dimensions():
    torch = pytest.importorskip("torch")
    from clear_lewm.dinowm_gd import _terminal_latent_mean_cost

    predicted = torch.ones(2, 3, 4, 192)
    goal = torch.zeros(2, 4, 192)
    cost = _terminal_latent_mean_cost(predicted, goal)
    assert cost.shape == (2, 3)
    assert torch.equal(cost, torch.ones(2, 3))


def test_dinowm_manual_sgd_matches_paper_update():
    torch = pytest.importorskip("torch")
    from clear_lewm.dinowm_gd import DINOWMGDPlanner

    class QuadraticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def get_cost(self, info_dict, actions):
            return (actions - 1.0).pow(2).mean(dim=(2, 3))

    model = QuadraticModel()
    planner = DINOWMGDPlanner(
        model,
        n_steps=1,
        batch_size=2,
        action_noise=0.0,
        device="cpu",
        seed=7,
        lr=0.1,
    )
    planner.configure(
        action_space=SimpleNamespace(shape=(2, 1)),
        n_envs=2,
        config=SimpleNamespace(horizon=2, action_block=1),
    )
    expected_initial = torch.randn(2, 2, 1, generator=torch.Generator().manual_seed(7))
    expected = expected_initial - 0.1 * (expected_initial - 1.0)
    result = planner.solve({"pixels": torch.zeros(2, 1)})
    assert torch.allclose(result["actions"], expected)


def test_dinowm_profile_uses_published_defaults():
    pytest.importorskip("torch")
    from clear_lewm.dinowm_gd import solver_config

    assert solver_config() == {
        "_target_": "clear_lewm.dinowm_gd.DINOWMGDPlanner",
        "model": "???",
        "n_steps": 1000,
        "batch_size": 1,
        "num_samples": 1,
        "action_noise": 0.003,
        "device": "cuda",
        "seed": "${seed}",
        "lr": 1.0,
    }


def test_non_cem_planner_rejects_cem_only_options(tmp_path):
    common = {
        "manifest_path": tmp_path / "missing.json",
        "policy": "random",
        "output": tmp_path / "out.json",
        "planner": "adam",
    }
    with pytest.raises(ValueError, match="world-model planning"):
        evaluate_manifest(**common, inference_mode="direct")
    with pytest.raises(ValueError, match="only supported by the CEM"):
        evaluate_manifest(**common, topk=30)


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
