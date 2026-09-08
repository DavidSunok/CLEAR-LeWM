"""Opt-in real-checkpoint GPU regression for the experimental planners.

Set ``CLEAR_LEWM_RUN_GPU_REGRESSION=1``, ``CLEAR_LEWM_GPU_DATASET``, and
``CLEAR_LEWM_GPU_CACHE_DIR`` before running this module with pytest.
"""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path

import pytest

from clear_lewm.runner import evaluate_manifest

RUN_GPU_REGRESSION = os.environ.get("CLEAR_LEWM_RUN_GPU_REGRESSION") == "1"


def _require_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if value is None:
        pytest.fail(f"{variable} must be set for the GPU regression")
    path = Path(value).resolve()
    if not path.exists():
        pytest.fail(f"{variable} does not exist: {path}")
    return path


def _two_pair_manifest(tmp_path: Path) -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "v0.8"
        / "pusht"
        / "moderate-seed0-n100.json"
    )
    manifest = json.loads(source.read_text())
    manifest["pairs"] = manifest["pairs"][:2]
    manifest["statistics"]["num_eval"] = 2
    output = tmp_path / "pusht-moderate-seed0-n2.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def _assert_same_core_result(first: dict, second: dict) -> None:
    for key in (
        "task",
        "protocol",
        "policy",
        "checkpoint",
        "policy_seed",
        "dataset_name",
        "dataset_fingerprint",
        "criterion",
        "metrics",
        "episode_successes",
        "raw_world_metrics",
    ):
        assert first[key] == second[key], key


@pytest.mark.skipif(
    not RUN_GPU_REGRESSION,
    reason="set CLEAR_LEWM_RUN_GPU_REGRESSION=1 for the real-checkpoint smoke",
)
def test_official_pusht_checkpoint_planners_on_stable_worldmodel_010(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_worldmodel")
    assert torch.cuda.is_available(), "the real-checkpoint regression requires CUDA"
    assert version("stable-worldmodel") == "0.1.0"

    dataset = _require_path("CLEAR_LEWM_GPU_DATASET")
    cache_dir = _require_path("CLEAR_LEWM_GPU_CACHE_DIR")
    upstream = Path(__file__).resolve().parents[1] / "third_party" / "le-wm"
    manifest = _two_pair_manifest(tmp_path)
    policy = os.environ.get("CLEAR_LEWM_GPU_POLICY", "official/pusht")
    common = {
        "manifest_path": manifest,
        "policy": policy,
        "cache_dir": cache_dir,
        "dataset_path": dataset,
        "upstream_dir": upstream,
        "policy_seed": 0,
        "actor_warmstart": False,
        "solver_batch_size": 1,
        "cpu_threads": 1,
        "matmul_precision": "highest",
        "strict_checkpoint": True,
    }

    default_cem = evaluate_manifest(
        **common,
        output=tmp_path / "cem-default.json",
        num_samples=4,
        n_steps=2,
        topk=2,
    )
    explicit_cem = evaluate_manifest(
        **common,
        output=tmp_path / "cem-explicit.json",
        planner="cem",
        num_samples=4,
        n_steps=2,
        topk=2,
    )
    _assert_same_core_result(default_cem, explicit_cem)

    adam = evaluate_manifest(
        **common,
        output=tmp_path / "adam.json",
        planner="adam",
        num_samples=2,
        n_steps=2,
    )
    assert adam["inference"]["solver_target"] == (
        "clear_lewm.adam.DeviceSafeGradientSolver"
    )
    assert adam["solver"]["source"]["version"] == "0.1.0"
    assert adam["environment"]["execution"]["status"] == "verified"
    assert adam["environment"]["accelerator"] is not None

    dino_common = dict(common)
    dino_common.pop("actor_warmstart")
    dino_runs = [
        evaluate_manifest(
            **dino_common,
            output=tmp_path / f"dinowm-gd-{run}.json",
            planner="dinowm-gd",
            num_samples=1,
            n_steps=10,
        )
        for run in range(2)
    ]
    _assert_same_core_result(*dino_runs)
    for result in dino_runs:
        diagnostics = result["solver"]["diagnostics"]
        assert diagnostics["solve_calls"] > 0
        assert diagnostics["finite_actions"] is True
        assert diagnostics["finite_costs"] is True
        assert diagnostics["total_solve_time_s"] > 0.0
        assert result["solver"]["initialization"] == "random-normal"
        assert result["solver"]["actor_prior_initialization"] is False
        assert result["inference"]["actor_warmstart_requested"] is False
        assert result["inference"]["actor_warmstart_effective"] is False
        assert result["inference"]["mode"] == "pure-dinowm-gd"
        assert result["environment"]["execution"]["status"] == "verified"
        assert result["environment"]["accelerator"] is not None
