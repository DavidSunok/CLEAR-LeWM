from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clear_lewm.submissions import SubmissionValidationError, validate_submission

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "example-method" / "v1"
    results = bundle / "results"
    results.mkdir(parents=True)
    result_path = results / "pusht-strict.json"
    result_path.write_bytes(
        (ROOT / "results/v0.5/pusht-strict-official-lewm-seed42-n100.json").read_bytes()
    )
    submission = {
        "schema_version": "clear-lewm-submission-v1",
        "method": {
            "name": "Example Method",
            "repository": "https://github.com/example/method",
            "revision": "0123456789abcdef",
            "license": "MIT",
        },
        "contact": {"github": "example"},
        "benchmark": {
            "version": "v0.5",
            "training_data_track": "standard-data",
            "training_data": {
                "description": "Canonical LeWM expert training datasets",
                "source": "LeWM public release",
                "revision": "public-v1",
                "license": "upstream terms",
            },
        },
        "verification": {"requested": "self-reported"},
        "results": [
            {
                "task": "pusht",
                "protocol": "strict",
                "path": "results/pusht-strict.json",
                "sha256": _sha256(result_path),
            }
        ],
    }
    path = bundle / "submission.json"
    path.write_text(json.dumps(submission))
    return path


def _v05_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "example-v05" / "v1"
    results = bundle / "results"
    results.mkdir(parents=True)
    result_path = results / "pusht-moderate.json"
    result_path.write_bytes(
        (
            ROOT / "results/v0.5/pusht-moderate-official-lewm-seed42-n100.json"
        ).read_bytes()
    )
    submission = {
        "schema_version": "clear-lewm-submission-v1",
        "method": {
            "name": "Example v0.5 Method",
            "repository": "https://github.com/example/method",
            "revision": "0123456789abcdef",
            "license": "MIT",
        },
        "contact": {"github": "example"},
        "benchmark": {
            "version": "v0.5",
            "training_data_track": "standard-data",
            "training_data": {
                "description": "Canonical LeWM expert training datasets",
                "source": "LeWM public release",
                "revision": "public-v1",
                "license": "upstream terms",
            },
        },
        "verification": {"requested": "self-reported"},
        "results": [
            {
                "task": "pusht",
                "protocol": "moderate",
                "path": "results/pusht-moderate.json",
                "sha256": _sha256(result_path),
            }
        ],
    }
    path = bundle / "submission.json"
    path.write_text(json.dumps(submission))
    return path


def test_valid_submission_checks_canonical_manifest_and_trace(tmp_path):
    report = validate_submission(_bundle(tmp_path), repo_root=ROOT)
    assert report["status"] == "valid"
    assert report["results"][0]["success_rate_percent"] == 71.0
    assert report["complete_four_task_matrix"] is False
    assert "cube/moderate" in report["missing_results"]


def test_v05_submission_accepts_moderate_and_uses_v05_random_floor(tmp_path):
    report = validate_submission(_v05_bundle(tmp_path), repo_root=ROOT)
    assert report["status"] == "valid"
    assert report["benchmark_version"] == "v0.5"
    assert report["results"][0]["success_rate_percent"] == 88.0
    assert report["results"][0]["random_success_rate_percent"] == 3.0
    assert "pusht/strict" in report["missing_results"]


def test_submission_rejects_tampered_result_hash(tmp_path):
    path = _bundle(tmp_path)
    submission = json.loads(path.read_text())
    submission["results"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(submission))
    with pytest.raises(SubmissionValidationError, match="SHA-256"):
        validate_submission(path, repo_root=ROOT)


def test_submission_rejects_noncanonical_manifest(tmp_path):
    path = _bundle(tmp_path)
    submission = json.loads(path.read_text())
    result_path = path.parent / submission["results"][0]["path"]
    result = json.loads(result_path.read_text())
    result["manifest_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result))
    submission["results"][0]["sha256"] = _sha256(result_path)
    path.write_text(json.dumps(submission))
    with pytest.raises(SubmissionValidationError, match="canonical v0.5 manifest"):
        validate_submission(path, repo_root=ROOT)


def test_reproducible_submission_requires_checkpoint_url(tmp_path):
    path = _bundle(tmp_path)
    submission = json.loads(path.read_text())
    submission["verification"] = {
        "requested": "reproducible",
        "command": "clear-lewm evaluate ...",
    }
    path.write_text(json.dumps(submission))
    with pytest.raises(SubmissionValidationError, match="checkpoint"):
        validate_submission(path, repo_root=ROOT)
