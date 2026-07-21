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
