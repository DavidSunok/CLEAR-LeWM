from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .metrics import bootstrap_ci

CORE_IDENTITY_FIELDS = {
    "task": ("task",),
    "protocol": ("protocol",),
    "manifest_sha256": ("manifest_sha256",),
    "policy_seed": ("policy_seed",),
    "dataset_fingerprint": ("dataset_fingerprint",),
    "checkpoint_runtime_sha256": ("checkpoint", "runtime_sha256"),
    "checkpoint_config_sha256": ("checkpoint", "config_sha256"),
    "solver": ("solver",),
    "inference": ("inference",),
    "cpu_threads": ("runtime", "cpu_threads"),
    "float32_matmul_precision": (
        "runtime",
        "float32_matmul_precision",
    ),
}

ENVIRONMENT_FIELDS = {
    "physics_fingerprint": ("environment", "physics_fingerprint"),
    "execution_fingerprint": ("environment", "execution_fingerprint"),
    "numerics_fingerprint": ("environment", "numerics_fingerprint"),
}


def _load_result(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"CLEAR result must be a JSON object: {path}")
    if payload.get("schema_version") != "clear-lewm-result-v1":
        raise ValueError(f"Unsupported CLEAR result schema: {path}")
    return payload


def _nested(payload: dict, path: tuple[str, ...]):
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _identity_checks(reference: dict, candidate: dict) -> dict:
    fields = {**CORE_IDENTITY_FIELDS, **ENVIRONMENT_FIELDS}
    return {
        label: {
            "match": _nested(reference, path) == _nested(candidate, path),
            "reference": _nested(reference, path),
            "candidate": _nested(candidate, path),
        }
        for label, path in fields.items()
    }


def _exact_mcnemar_pvalue(reference_only: int, candidate_only: int) -> float:
    discordant = reference_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(reference_only, candidate_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_outcomes(reference: dict, candidate: dict) -> dict:
    reference_trace = np.asarray(reference.get("episode_successes"), dtype=bool)
    candidate_trace = np.asarray(candidate.get("episode_successes"), dtype=bool)
    if reference_trace.ndim != 1 or candidate_trace.ndim != 1:
        raise ValueError("episode_successes must be one-dimensional in both results")
    if len(reference_trace) == 0 or len(reference_trace) != len(candidate_trace):
        raise ValueError("Results must contain equally sized, non-empty episode traces")

    both_success = int(np.logical_and(reference_trace, candidate_trace).sum())
    reference_only = int(np.logical_and(reference_trace, ~candidate_trace).sum())
    candidate_only = int(np.logical_and(~reference_trace, candidate_trace).sum())
    both_fail = int(np.logical_and(~reference_trace, ~candidate_trace).sum())
    paired_delta = candidate_trace.astype(float) - reference_trace.astype(float)
    low, high = bootstrap_ci(paired_delta, seed=0, samples=10_000)
    episodes = len(reference_trace)

    return {
        "episodes": episodes,
        "reference_success_rate_percent": float(reference_trace.mean() * 100.0),
        "candidate_success_rate_percent": float(candidate_trace.mean() * 100.0),
        "candidate_minus_reference_pp": float(paired_delta.mean() * 100.0),
        "paired_delta_ci95_pp": [low * 100.0, high * 100.0],
        "both_success": both_success,
        "reference_only": reference_only,
        "candidate_only": candidate_only,
        "both_fail": both_fail,
        "trace_agreement_percent": float(
            (both_success + both_fail) / episodes * 100.0
        ),
        "exact_mcnemar_pvalue": _exact_mcnemar_pvalue(
            reference_only, candidate_only
        ),
    }


def compare_reproduction_results(
    reference_path: str | Path,
    candidate_path: str | Path,
) -> dict:
    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)
    reference = _load_result(reference_path)
    candidate = _load_result(candidate_path)
    checks = _identity_checks(reference, candidate)
    core_match = all(checks[label]["match"] for label in CORE_IDENTITY_FIELDS)
    physics_match = checks["physics_fingerprint"]["match"]
    execution_match = checks["execution_fingerprint"]["match"]
    numerics_match = checks["numerics_fingerprint"]["match"]
    paired = _paired_outcomes(reference, candidate)

    if not core_match or not physics_match or not execution_match:
        classification = "incompatible"
        expectation = "Results do not share the fixed evaluation identity."
    elif not numerics_match:
        classification = "cross-numerics-sensitivity"
        expectation = (
            "The fixed protocol matches, but exact episode reproduction is not "
            "claimed across numerical fingerprints."
        )
    elif paired["trace_agreement_percent"] == 100.0:
        classification = "exact-reproduction"
        expectation = "Fixed identity and all episode outcomes match exactly."
    else:
        classification = "same-runtime-drift"
        expectation = (
            "Numerical identity matches but episode outcomes drifted; investigate "
            "unrecorded state or nondeterministic execution."
        )

    return {
        "schema_version": "clear-lewm-reproduction-comparison-v1",
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "classification": classification,
        "expectation": expectation,
        "identity_checks": checks,
        "paired_outcomes": paired,
    }
