from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clear_lewm.protocols import protocol_from_dict

EXPECTED_SR = {
    "pusht": {"official": (89.0, 7.0), "moderate": (74.0, 0.0), "strict": (42.0, 0.0)},
    "reacher": {
        "official": (87.0, 16.0),
        "moderate": (63.0, 6.0),
        "strict": (22.0, 0.0),
    },
    "tworoom": {
        "official": (85.0, 30.0),
        "moderate": (70.0, 17.0),
        "strict": (41.0, 1.0),
    },
    "cube": {"official": (62.0, 47.0), "moderate": (36.0, 3.0), "strict": (17.0, 2.0)},
}

EXPECTED_V05_MODERATE_SR = {
    "pusht": {0: (84.0, 4.0), 1: (87.0, 5.0), 42: (88.0, 3.0)},
    "cube": {0: (46.0, 10.0), 1: (54.0, 22.0), 42: (51.0, 15.0)},
    "reacher": {0: (51.0, 5.0), 1: (47.0, 3.0), 42: (40.0, 5.0)},
    "tworoom": {0: (87.0, 8.0), 1: (84.0, 6.0), 42: (81.0, 6.0)},
}

EXPECTED_V05_STRICT_SR = {
    "pusht": {0: (66.0, 4.0), 1: (74.0, 4.0), 42: (71.0, 7.0)},
    "cube": {0: (28.0, 3.0), 1: (26.0, 7.0), 42: (25.0, 8.0)},
    "reacher": {0: (41.0, 1.0), 1: (37.0, 7.0), 42: (51.0, 7.0)},
    "tworoom": {0: (61.0, 5.0), 1: (57.0, 0.0), 42: (57.0, 0.0)},
}


def test_primary_reference_results_match_manifests_and_provenance():
    root = Path(__file__).resolve().parents[1]
    for task, tiers in EXPECTED_SR.items():
        for tier, (model_sr, random_sr) in tiers.items():
            manifest_path = root / "manifests" / task / f"{tier}-seed42-n100.json"
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            for policy, expected_sr in (
                ("random", random_sr),
                ("official-lewm", model_sr),
            ):
                result_path = (
                    root
                    / "results"
                    / "reference"
                    / f"{task}-{tier}-{policy}-seed42-n100.json"
                )
                result = json.loads(result_path.read_text())
                spec = protocol_from_dict(result["protocol"])
                assert result["manifest_sha256"] == manifest_hash
                assert result["criterion"]["sustained_steps"] == spec.hold_steps(task)
                assert result["metrics"]["success_rate_percent"] == pytest.approx(
                    expected_sr
                )
                assert len(result["episode_successes"]) == 100
                if policy == "random":
                    assert result.get("checkpoint") is None
                else:
                    source = result["checkpoint"]["source"]
                    assert source["tensors_loaded"] == 303
                    assert len(source["source_weights_sha256"]) == 64


def test_v05_moderate_results_are_paired_and_strictly_loaded():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "checkpoints" / "official-v0.5.json").read_text())[
        "models"
    ]
    for task, seeds in EXPECTED_V05_MODERATE_SR.items():
        for seed, (model_sr, random_sr) in seeds.items():
            manifest_path = (
                root / "manifests" / "v0.5" / task / f"moderate-seed{seed}-n100.json"
            )
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            random_result = json.loads(
                (
                    root
                    / "results"
                    / "v0.5"
                    / f"{task}-moderate-random-seed{seed}-n100.json"
                ).read_text()
            )
            model_result = json.loads(
                (
                    root
                    / "results"
                    / "v0.5"
                    / f"{task}-moderate-official-lewm-seed{seed}-n100.json"
                ).read_text()
            )
            assert random_result["manifest_sha256"] == manifest_hash
            assert model_result["manifest_sha256"] == manifest_hash
            assert random_result["metrics"]["success_rate_percent"] == random_sr
            assert model_result["metrics"]["success_rate_percent"] == model_sr
            assert model_result["metrics"]["random_success_rate_percent"] == random_sr
            assert model_result["solver"] == {
                "batch_size": 1,
                "n_steps": 30,
                "num_samples": 300,
                "topk": 30,
            }
            audit = model_result["checkpoint"]["state_dict_audit"]
            assert audit["missing_keys"] == []
            assert audit["unexpected_keys"] == []
            assert (
                model_result["checkpoint"]["runtime_sha256"]
                == registry[task]["runtime_weights_sha256"]
            )
            if task == "tworoom":
                assert model_result["topology"]["invalid_routes"] == 0
                assert random_result["topology"]["invalid_routes"] == 0


def test_v05_strict_three_seed_results_match_manifests_and_provenance():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "checkpoints" / "official-v0.5.json").read_text())[
        "models"
    ]
    for task, seeds in EXPECTED_V05_STRICT_SR.items():
        for seed, (model_sr, random_sr) in seeds.items():
            manifest_path = (
                root / "manifests" / "v0.5" / task / f"strict-seed{seed}-n100.json"
            )
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            for policy, expected_sr in (
                ("random", random_sr),
                ("official-lewm", model_sr),
            ):
                result = json.loads(
                    (
                        root
                        / "results"
                        / "v0.5"
                        / f"{task}-strict-{policy}-seed{seed}-n100.json"
                    ).read_text()
                )
                assert result["manifest_sha256"] == manifest_hash
                assert result["metrics"]["success_rate_percent"] == pytest.approx(
                    expected_sr
                )
                assert len(result["episode_successes"]) == 100
                assert result["criterion"]["sustained_steps"] == protocol_from_dict(
                    result["protocol"]
                ).hold_steps(task)
                if task == "tworoom":
                    assert result["topology"]["invalid_routes"] == 0
                if policy == "random":
                    assert result.get("checkpoint") is None
                else:
                    checkpoint = result["checkpoint"]
                    assert checkpoint["state_dict_audit"]["missing_keys"] == []
                    assert checkpoint["state_dict_audit"]["unexpected_keys"] == []
                    assert (
                        checkpoint["runtime_sha256"]
                        == registry[task]["runtime_weights_sha256"]
                    )
