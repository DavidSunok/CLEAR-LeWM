from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from .protocols import TASKS, protocol_from_dict

SUBMISSION_SCHEMA = "clear-lewm-submission-v1"
RESULT_SCHEMA = "clear-lewm-result-v1"
BENCHMARK_VERSIONS = ("v0.3", "v0.5")
TRAINING_DATA_TRACKS = ("standard-data", "reduced-data", "external-data")
VERIFICATION_REQUESTS = ("self-reported", "reproducible")
PRIMARY_PROTOCOLS = ("moderate", "strict")


class SubmissionValidationError(ValueError):
    pass


def _load_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        message = f"Cannot read {label} {path}: {error}"
        raise SubmissionValidationError(message) from error
    if not isinstance(payload, dict):
        raise SubmissionValidationError(f"{label} must contain a JSON object: {path}")
    return payload


def _required_object(payload: dict, key: str, label: str) -> dict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SubmissionValidationError(f"{label}.{key} must be a JSON object")
    return value


def _required_string(payload: dict, key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SubmissionValidationError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _https_url(payload: dict, key: str, label: str) -> str:
    value = _required_string(payload, key, label)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise SubmissionValidationError(f"{label}.{key} must be an HTTPS URL")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _resolve_bundle_path(bundle: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SubmissionValidationError("Each result path must be a non-empty string")
    candidate = (bundle / relative).resolve()
    try:
        candidate.relative_to(bundle.resolve())
    except ValueError as error:
        raise SubmissionValidationError(
            f"Result path must stay inside the submission bundle: {relative}"
        ) from error
    if candidate.suffix != ".json":
        raise SubmissionValidationError(f"Result must be JSON: {relative}")
    if not candidate.is_file():
        raise SubmissionValidationError(f"Result file does not exist: {relative}")
    return candidate


def _canonical_manifest(
    repo_root: Path, version: str, task: str, protocol: str
) -> tuple[Path, dict]:
    path = repo_root / "manifests" / version / task / f"{protocol}-seed42-n100.json"
    if not path.is_file():
        raise SubmissionValidationError(f"Canonical manifest is missing: {path}")
    return path, _load_object(path, "manifest")


def _canonical_random_rate(
    repo_root: Path, version: str, task: str, protocol: str
) -> float:
    path = (
        repo_root / "results" / version / f"{task}-{protocol}-random-seed42-n100.json"
    )
    payload = _load_object(path, "canonical random result")
    try:
        return float(payload["metrics"]["success_rate_percent"])
    except (KeyError, TypeError, ValueError) as error:
        raise SubmissionValidationError(
            f"Canonical random result has no valid success rate: {path}"
        ) from error


def _validate_environment(result: dict, label: str) -> None:
    environment = _required_object(result, "environment", label)
    for key in ("physics_fingerprint", "numerics_fingerprint", "execution_fingerprint"):
        if not _is_sha256(environment.get(key)):
            raise SubmissionValidationError(f"{label}.environment.{key} is invalid")


def _validate_checkpoint(result: dict, label: str) -> None:
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise SubmissionValidationError(
            f"{label}.checkpoint must identify the evaluated model"
        )
    if not _is_sha256(checkpoint.get("runtime_sha256")):
        raise SubmissionValidationError(
            f"{label}.checkpoint.runtime_sha256 must be a SHA-256"
        )
    source = _required_object(checkpoint, "source", f"{label}.checkpoint")
    _required_string(source, "repository", f"{label}.checkpoint.source")
    _required_string(source, "revision", f"{label}.checkpoint.source")


def _validate_result(
    result: dict,
    *,
    result_path: Path,
    repo_root: Path,
    version: str,
    expected_task: str,
    expected_protocol: str,
) -> dict:
    label = str(result_path)
    if result.get("schema_version") != RESULT_SCHEMA:
        raise SubmissionValidationError(f"Unsupported result schema in {label}")
    if result.get("task") != expected_task:
        raise SubmissionValidationError(
            f"Result task does not match submission: {label}"
        )
    protocol = _required_object(result, "protocol", label)
    if protocol.get("name") != expected_protocol:
        raise SubmissionValidationError(
            f"Result protocol does not match submission: {label}"
        )

    manifest_path, manifest = _canonical_manifest(
        repo_root, version, expected_task, expected_protocol
    )
    manifest_sha256 = _sha256(manifest_path)
    if result.get("manifest_sha256") != manifest_sha256:
        raise SubmissionValidationError(
            f"Result does not use the canonical {version} manifest: {label}"
        )
    if protocol != manifest.get("protocol"):
        raise SubmissionValidationError(f"Result protocol payload drifted: {label}")
    if result.get("policy_seed") != manifest.get("policy_seed"):
        raise SubmissionValidationError(f"Result policy seed drifted: {label}")
    if result.get("dataset_fingerprint") != manifest.get("dataset", {}).get(
        "fingerprint"
    ):
        raise SubmissionValidationError(f"Result evaluation dataset drifted: {label}")

    trace = result.get("episode_successes")
    expected_episodes = len(manifest.get("pairs", []))
    if (
        not isinstance(trace, list)
        or len(trace) != expected_episodes
        or any(type(outcome) is not bool for outcome in trace)
    ):
        raise SubmissionValidationError(
            f"Result must contain {expected_episodes} boolean episode outcomes: {label}"
        )
    success_rate = 100.0 * sum(trace) / len(trace)
    metrics = _required_object(result, "metrics", label)
    if metrics.get("episodes") != expected_episodes or not math.isclose(
        float(metrics.get("success_rate_percent", math.nan)),
        success_rate,
        abs_tol=1e-9,
    ):
        raise SubmissionValidationError(
            f"Result metrics do not match its trace: {label}"
        )

    random_rate = _canonical_random_rate(
        repo_root, version, expected_task, expected_protocol
    )
    if not math.isclose(
        float(metrics.get("random_success_rate_percent", math.nan)),
        random_rate,
        abs_tol=1e-9,
    ):
        raise SubmissionValidationError(
            f"Result does not use the canonical paired random rate: {label}"
        )
    if not math.isclose(
        float(metrics.get("excess_over_random_pp", math.nan)),
        success_rate - random_rate,
        abs_tol=1e-9,
    ):
        raise SubmissionValidationError(
            f"Result excess over random is invalid: {label}"
        )

    criterion = _required_object(result, "criterion", label)
    expected_hold = protocol_from_dict(protocol).hold_steps(expected_task)
    if criterion.get("sustained_steps") != expected_hold:
        raise SubmissionValidationError(f"Result hold criterion drifted: {label}")
    _required_object(result, "solver", label)
    _required_object(result, "inference", label)
    _validate_environment(result, label)
    _validate_checkpoint(result, label)

    if expected_task == "tworoom":
        topology = _required_object(result, "topology", label)
        if topology.get("invalid_routes") != 0:
            raise SubmissionValidationError(
                f"TwoRoom submission contains invalid routes: {label}"
            )

    return {
        "task": expected_task,
        "protocol": expected_protocol,
        "success_rate_percent": success_rate,
        "random_success_rate_percent": random_rate,
        "excess_over_random_pp": success_rate - random_rate,
    }


def _find_repo_root(submission_path: Path) -> Path:
    for candidate in (submission_path.parent, *submission_path.parents):
        if any(
            (candidate / "manifests" / version).is_dir()
            for version in BENCHMARK_VERSIONS
        ):
            return candidate
    raise SubmissionValidationError(
        "Cannot locate the CLEAR-LeWM repository; pass repo_root explicitly"
    )


def validate_submission(
    submission_path: str | Path, *, repo_root: str | Path | None = None
) -> dict:
    submission_path = Path(submission_path).resolve()
    submission = _load_object(submission_path, "submission")
    bundle = submission_path.parent
    root = Path(repo_root).resolve() if repo_root else _find_repo_root(submission_path)

    if submission.get("schema_version") != SUBMISSION_SCHEMA:
        raise SubmissionValidationError(
            f"submission.schema_version must be {SUBMISSION_SCHEMA!r}"
        )

    method = _required_object(submission, "method", "submission")
    method_name = _required_string(method, "name", "submission.method")
    _https_url(method, "repository", "submission.method")
    _required_string(method, "revision", "submission.method")
    _required_string(method, "license", "submission.method")

    contact = _required_object(submission, "contact", "submission")
    _required_string(contact, "github", "submission.contact")

    benchmark = _required_object(submission, "benchmark", "submission")
    version = _required_string(benchmark, "version", "submission.benchmark")
    if version not in BENCHMARK_VERSIONS:
        raise SubmissionValidationError(f"Unsupported benchmark version: {version}")
    accepted_protocols = ("moderate",) if version == "v0.5" else PRIMARY_PROTOCOLS
    track = _required_string(benchmark, "training_data_track", "submission.benchmark")
    if track not in TRAINING_DATA_TRACKS:
        raise SubmissionValidationError(f"Unsupported training-data track: {track}")
    training_data = _required_object(benchmark, "training_data", "submission.benchmark")
    for key in ("description", "source", "revision", "license"):
        _required_string(training_data, key, "submission.benchmark.training_data")
    if track == "reduced-data":
        fraction = training_data.get("fraction")
        if not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= 1:
            raise SubmissionValidationError(
                "Reduced-data submissions require training_data.fraction in (0, 1]"
            )

    verification = _required_object(submission, "verification", "submission")
    requested = _required_string(verification, "requested", "submission.verification")
    if requested not in VERIFICATION_REQUESTS:
        raise SubmissionValidationError(
            f"Unsupported verification request: {requested}"
        )
    if requested == "reproducible":
        _https_url(verification, "checkpoint", "submission.verification")
        _required_string(verification, "command", "submission.verification")

    entries = submission.get("results")
    if not isinstance(entries, list) or not entries:
        raise SubmissionValidationError("submission.results must be a non-empty list")

    seen: set[tuple[str, str]] = set()
    summaries = []
    for index, entry in enumerate(entries):
        label = f"submission.results[{index}]"
        if not isinstance(entry, dict):
            raise SubmissionValidationError(f"{label} must be a JSON object")
        task = _required_string(entry, "task", label)
        protocol = _required_string(entry, "protocol", label)
        if task not in TASKS:
            raise SubmissionValidationError(f"Unsupported task in {label}: {task}")
        if protocol not in accepted_protocols:
            raise SubmissionValidationError(
                f"Protocol {protocol!r} is not accepted for {version}: {label}"
            )
        identity = (task, protocol)
        if identity in seen:
            raise SubmissionValidationError(f"Duplicate result: {task}/{protocol}")
        seen.add(identity)

        result_path = _resolve_bundle_path(bundle, entry.get("path"))
        declared_hash = entry.get("sha256")
        actual_hash = _sha256(result_path)
        if not _is_sha256(declared_hash) or declared_hash != actual_hash:
            raise SubmissionValidationError(
                f"Result SHA-256 does not match {result_path.relative_to(bundle)}"
            )
        result = _load_object(result_path, "result")
        summaries.append(
            _validate_result(
                result,
                result_path=result_path,
                repo_root=root,
                version=version,
                expected_task=task,
                expected_protocol=protocol,
            )
        )

    full_matrix = {(task, mode) for task in TASKS for mode in accepted_protocols}
    missing = sorted(full_matrix - seen)
    return {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "valid",
        "method": method_name,
        "benchmark_version": version,
        "training_data_track": track,
        "verification_requested": requested,
        "results": summaries,
        "complete_four_task_matrix": not missing,
        "missing_results": [f"{task}/{protocol}" for task, protocol in missing],
    }
