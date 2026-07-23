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


def test_moderate_tworoom_rejects_contaminated_source_windows(tworoom_dataset):
    manifest = generate_manifest(
        tworoom_dataset,
        task="tworoom",
        protocol="moderate",
        num_eval=2,
        seed=42,
    )
    assert manifest["statistics"]["pairs_after_filters"] == 14
    assert manifest["statistics"]["unique_episodes"] == 2
    assert all(pair["cross_room"] is True for pair in manifest["pairs"])
    assert all(pair["source_window_clean"] is True for pair in manifest["pairs"])


def test_strict_reacher_filters_by_endpoint_distance(reacher_dataset):
    manifest = generate_manifest(
        reacher_dataset,
        task="reacher",
        protocol="strict",
        num_eval=2,
        seed=42,
    )
    assert all(not pair["initial_success"] for pair in manifest["pairs"])
    assert all(pair["endpoint_distance_m"] > 0.01 for pair in manifest["pairs"])
