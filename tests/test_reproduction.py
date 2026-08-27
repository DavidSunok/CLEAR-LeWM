from __future__ import annotations

import copy
import json

from clear_lewm.reproduction import compare_reproduction_results


def _result() -> dict:
    return {
        "schema_version": "clear-lewm-result-v1",
        "task": "tworoom",
        "protocol": {"name": "strict"},
        "manifest_sha256": "manifest",
        "policy_seed": 0,
        "dataset_fingerprint": {"kind": "metadata-sha256", "value": "data"},
        "checkpoint": {
            "runtime_sha256": "weights",
            "config_sha256": "config",
        },
        "solver": {"batch_size": 1, "num_samples": 300, "n_steps": 30},
        "inference": {"mode": "pure-cem"},
        "runtime": {
            "cpu_threads": 1,
            "float32_matmul_precision": "highest",
        },
        "environment": {
            "physics_fingerprint": "physics",
            "execution_fingerprint": "execution",
            "numerics_fingerprint": "h200",
        },
        "episode_successes": [True, True, False, False],
    }


def _compare(tmp_path, reference: dict, candidate: dict) -> dict:
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    reference_path.write_text(json.dumps(reference))
    candidate_path.write_text(json.dumps(candidate))
    return compare_reproduction_results(reference_path, candidate_path)


def test_exact_reproduction_requires_trace_identity(tmp_path):
    reference = _result()
    report = _compare(tmp_path, reference, copy.deepcopy(reference))
    assert report["classification"] == "exact-reproduction"
    assert report["paired_outcomes"]["trace_agreement_percent"] == 100.0


def test_cross_numerics_is_classified_as_sensitivity_run(tmp_path):
    reference = _result()
    candidate = copy.deepcopy(reference)
    candidate["environment"]["numerics_fingerprint"] = "ada"
    candidate["episode_successes"] = [True, False, True, False]

    report = _compare(tmp_path, reference, candidate)
    assert report["classification"] == "cross-numerics-sensitivity"
    paired = report["paired_outcomes"]
    assert paired["reference_only"] == 1
    assert paired["candidate_only"] == 1
    assert paired["exact_mcnemar_pvalue"] == 1.0


def test_same_runtime_trace_drift_is_an_error_classification(tmp_path):
    reference = _result()
    candidate = copy.deepcopy(reference)
    candidate["episode_successes"][0] = False
    report = _compare(tmp_path, reference, candidate)
    assert report["classification"] == "same-runtime-drift"


def test_protocol_identity_mismatch_is_incompatible(tmp_path):
    reference = _result()
    candidate = copy.deepcopy(reference)
    candidate["manifest_sha256"] = "other"
    report = _compare(tmp_path, reference, candidate)
    assert report["classification"] == "incompatible"
