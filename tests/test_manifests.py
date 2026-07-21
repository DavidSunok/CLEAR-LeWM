from __future__ import annotations

from clear_lewm.audit import audit_dataset
from clear_lewm.manifests import generate_manifest


def test_audit_detects_initially_solved_cube_pairs(cube_dataset):
    report = audit_dataset(cube_dataset, task="cube", goal_offset=2)
    assert report["valid_pairs"] == 600
    assert report["initial_success_rate_percent"] == 50.0


def test_clear_manifest_is_heldout_balanced_and_nontrivial(cube_dataset):
    manifest = generate_manifest(
        cube_dataset,
        task="cube",
        protocol="clear-standard",
        num_eval=2,
        seed=42,
    )
    assert manifest["split"] == "heldout"
    assert manifest["statistics"]["initial_success_rate_percent"] in {0.0, 50.0, 100.0}
    assert manifest["statistics"]["unique_episodes"] == 2
    assert all(not pair["initial_success"] for pair in manifest["pairs"])
    assert all(pair["position_distance_m"] > 0.04 for pair in manifest["pairs"])


def test_manifest_generation_is_deterministic_except_timestamp(cube_dataset):
    kwargs = dict(
        dataset_path=cube_dataset,
        task="cube",
        protocol="official-compat",
        num_eval=10,
        seed=9,
    )
    first = generate_manifest(**kwargs)
    second = generate_manifest(**kwargs)
    first.pop("created_utc")
    second.pop("created_utc")
    assert first == second
