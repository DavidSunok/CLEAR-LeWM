from __future__ import annotations

from pathlib import Path

from clear_lewm.manifests import evaluation_contract, load_manifest
from clear_lewm.protocols import protocol_from_dict

ROOT = Path(__file__).resolve().parents[1]


def test_v09_manifests_change_initialization_without_resampling_pairs():
    for current_path in sorted((ROOT / "manifests" / "v0.9").glob("*/*.json")):
        legacy_path = (
            ROOT
            / "manifests"
            / "v0.8"
            / current_path.relative_to(ROOT / "manifests" / "v0.9")
        )
        current = load_manifest(current_path)
        legacy = load_manifest(legacy_path)

        assert evaluation_contract(current) == (
            "v0.9",
            {
                "mode": "rest-start",
                "history_source": "executed-rollout",
                "history_len": 3,
                "history_stride": 5,
            },
        )
        assert evaluation_contract(legacy) == (
            "v0.8",
            {
                "mode": "recorded-state",
                "history_source": "single-observation",
                "history_len": 1,
            },
        )
        assert current["dataset"] == legacy["dataset"]
        assert current["pairs"] == legacy["pairs"]
        assert protocol_from_dict(current["protocol"]) == protocol_from_dict(
            legacy["protocol"]
        )
