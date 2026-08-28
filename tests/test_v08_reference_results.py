from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "v0.8" / "runs"
TASKS = ("pusht", "cube", "reacher", "tworoom")
PROTOCOLS = ("moderate", "strict")
SEEDS = (0, 1, 42)

EXPECTED_OFFICIAL = {
    "pusht": {"moderate": (87, 87, 86), "strict": (65, 76, 71)},
    "cube": {"moderate": (46, 52, 53), "strict": (20, 27, 18)},
    "reacher": {"moderate": (79, 80, 80), "strict": (86, 87, 88)},
    "tworoom": {"moderate": (89, 81, 79), "strict": (50, 55, 49)},
}
EXPECTED_RANDOM = {
    "pusht": {"moderate": (4, 5, 3), "strict": (4, 4, 7)},
    "cube": {"moderate": (10, 22, 15), "strict": (3, 7, 8)},
    "reacher": {"moderate": (5, 8, 9), "strict": (1, 10, 13)},
    "tworoom": {"moderate": (8, 6, 6), "strict": (5, 0, 0)},
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_v08_sha256_inventory_covers_the_complete_matrix():
    checksum_file = ROOT / "results" / "v0.8" / "SHA256SUMS"
    entries = [
        line.split(maxsplit=1) for line in checksum_file.read_text().splitlines()
    ]
    assert len(entries) == 84
    for expected, relative in entries:
        path = ROOT / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    summary = load(ROOT / "results" / "v0.8" / "summary.json")
    assert summary["benchmark_version"] == "v0.8"
    assert summary["matrix"] == {
        "community_model_results": 36,
        "expected_result_files": 84,
        "official_model_results": 24,
        "paired_random_results": 24,
        "validated_result_files": 84,
    }
    assert hashlib.sha256(checksum_file.read_bytes()).hexdigest() == summary[
        "result_set_sha256"
    ]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_v08_official_and_random_results_are_paired(task: str, protocol: str):
    manifest_dir = ROOT / "manifests" / "v0.8" / task
    for index, seed in enumerate(SEEDS):
        manifest = manifest_dir / f"{protocol}-seed{seed}-n100.json"
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        model = load(RUNS / "official-lewm" / f"{task}-{protocol}-seed{seed}.json")
        random = load(RUNS / "random" / f"{task}-{protocol}-seed{seed}.json")

        for result in (model, random):
            assert result["benchmark_version"] == "v0.8"
            assert result["manifest_sha256"] == manifest_sha
            assert result["policy_seed"] == seed
            assert result["task"] == task
            assert result["protocol"]["name"] == protocol
            assert result["metrics"]["episodes"] == 100
            assert len(result["episode_successes"]) == 100
            assert result["environment"]["accelerator"]["name"] == (
                "NVIDIA GeForce RTX 4090"
            )

        assert model["metrics"]["success_rate_percent"] == pytest.approx(
            EXPECTED_OFFICIAL[task][protocol][index]
        )
        assert random["metrics"]["success_rate_percent"] == pytest.approx(
            EXPECTED_RANDOM[task][protocol][index]
        )
        assert model["metrics"]["random_success_rate_percent"] == pytest.approx(
            EXPECTED_RANDOM[task][protocol][index]
        )
        assert model["solver"] == {
            "batch_size": 1,
            "n_steps": 30,
            "num_samples": 300,
            "topk": 30,
        }
        assert model["inference"]["mode"] == "cem"
        assert model["inference"]["actor_warmstart_requested"] is False
        audit = model["checkpoint"]["state_dict_audit"]
        assert audit["strict_required"] is True
        assert audit["missing_keys"] == []
        assert audit["unexpected_keys"] == []
        assert model["runtime"]["reacher_task_termination_gate"] is (
            task == "reacher"
        )


def test_v08_related_checkpoint_matrix_is_complete_on_released_tasks():
    models = ("dinov2-no-proprio-lewm", "gcbc-joint-lewm")
    tasks = ("pusht", "cube", "tworoom")
    for model in models:
        files = sorted((RUNS / model).glob("*.json"))
        assert len(files) == 18
        checkpoint_hashes: dict[str, set[str]] = {task: set() for task in tasks}
        for path in files:
            result = load(path)
            assert result["benchmark_version"] == "v0.8"
            assert result["task"] in tasks
            assert result["policy_seed"] in SEEDS
            assert len(result["episode_successes"]) == 100
            assert result["inference"]["mode"] == "cem"
            assert result["inference"]["actor_warmstart_requested"] is False
            audit = result["checkpoint"]["state_dict_audit"]
            assert audit["strict_required"] is True
            assert audit["missing_keys"] == []
            assert audit["unexpected_keys"] == []
            checkpoint_hashes[result["task"]].add(
                result["checkpoint"]["runtime_sha256"]
            )
        assert all(len(hashes) == 1 for hashes in checkpoint_hashes.values())
