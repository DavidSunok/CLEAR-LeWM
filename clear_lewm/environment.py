from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

EVALUATION_PACKAGES = (
    "dm-control",
    "gymnasium",
    "hydra-core",
    "mujoco",
    "numpy",
    "ogbench",
    "omegaconf",
    "pygame",
    "pymunk",
    "scipy",
    "shapely",
    "stable-worldmodel",
    "torch",
    "torchvision",
)

TASK_ENVIRONMENT_MODULES = {
    "cube": "stable_worldmodel.envs.ogbench.cube_env",
    "pusht": "stable_worldmodel.envs.pusht.env",
    "reacher": "stable_worldmodel.envs.dmcontrol.custom_tasks.reacher",
    "tworoom": "stable_worldmodel.envs.two_room.env",
}

EVALUATION_RUNTIME_MODULES = {
    "world": "stable_worldmodel.world.world",
    "policy": "stable_worldmodel.policy",
    "cem": "stable_worldmodel.solver.cem",
    "checkpoint_loader": "stable_worldmodel.wm.utils",
}

CLEAR_RUNTIME_FILES = (
    "metrics.py",
    "protocols.py",
    "runner.py",
    "tasks.py",
    "tworoom_runtime.py",
)

# SHA-256 values from the stable-worldmodel 0.1.0 wheel published on PyPI.
OFFICIAL_RUNTIME_HASHES = {
    "0.1.0": {
        "world": "39318b81ed151d8556d8540f460a63eedaed0ce4b2211ce0af9f8200e7d83bde",
        "policy": "4967e7e3d5b20eb7a1d0b00e5d60fd701cce1c750ae7ec4a9b02529b9366db22",
        "cem": "a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b",
        "checkpoint_loader": (
            "48b8ea9fad66887d4f4fecc18d7e47b3d6fa9e7c3791c9769137ef20ab128424"
        ),
    }
}


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _fingerprint(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _module_source(module_name: str) -> dict:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError):
        spec = None
    if spec is None or spec.origin is None:
        return {"module": module_name, "file": None, "sha256": None}
    source = Path(spec.origin).resolve()
    return {
        "module": module_name,
        "file": source.name,
        "sha256": (
            hashlib.sha256(source.read_bytes()).hexdigest()
            if source.is_file()
            else None
        ),
    }


def _environment_source(task: str | None) -> dict | None:
    module_name = TASK_ENVIRONMENT_MODULES.get(task or "")
    return _module_source(module_name) if module_name is not None else None


def _clear_lewm_source_audit() -> dict:
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in CLEAR_RUNTIME_FILES
    }


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    return ",".join(sorted(versions)) or None


def _torch_numerical_controls(torch_module) -> dict:
    cuda_backend = getattr(torch_module.backends, "cuda", None)

    def cuda_flag(name: str):
        value = getattr(cuda_backend, name, None) if cuda_backend is not None else None
        return value() if callable(value) else None

    return {
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
        "deterministic_algorithms": (
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            torch_module.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": torch_module.backends.cudnn.benchmark,
        "cudnn_deterministic": torch_module.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch_module.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch_module.backends.cuda.matmul.allow_tf32,
        "flash_sdp_enabled": cuda_flag("flash_sdp_enabled"),
        "mem_efficient_sdp_enabled": cuda_flag("mem_efficient_sdp_enabled"),
        "math_sdp_enabled": cuda_flag("math_sdp_enabled"),
        "cudnn_sdp_enabled": cuda_flag("cudnn_sdp_enabled"),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_MODULE_LOADING",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
    }


def stable_worldmodel_source_audit() -> dict:
    package_version = _version("stable-worldmodel")
    sources = {
        name: _module_source(module_name)
        for name, module_name in EVALUATION_RUNTIME_MODULES.items()
    }
    expected = OFFICIAL_RUNTIME_HASHES.get(package_version or "")
    if expected is None:
        status = "unknown-version"
        mismatches = []
    else:
        mismatches = [
            name
            for name, expected_hash in expected.items()
            if sources[name]["sha256"] != expected_hash
        ]
        status = "verified" if not mismatches else "mismatch"
    return {
        "package": "stable-worldmodel",
        "version": package_version,
        "status": status,
        "mismatches": mismatches,
        "sources": sources,
        "expected_sha256": expected,
    }


def require_official_stable_worldmodel() -> dict:
    audit = stable_worldmodel_source_audit()
    if audit["status"] != "verified":
        details = ", ".join(audit["mismatches"]) or audit["status"]
        raise RuntimeError(
            "Unverified stable-worldmodel evaluation runtime: "
            f"version={audit['version']}, differences={details}. Reinstall the "
            "official wheel or pass --allow-modified-stable-worldmodel for an "
            "explicitly non-reference run."
        )
    return audit


def collect_environment(torch_module=None, task: str | None = None) -> dict:
    packages = {name: _version(name) for name in EVALUATION_PACKAGES}
    try:
        import mujoco

        mujoco_runtime = mujoco.mj_versionString()
    except (ImportError, OSError):
        mujoco_runtime = None

    physics = {
        "dm_control": packages["dm-control"],
        "gymnasium": packages["gymnasium"],
        "mujoco_package": packages["mujoco"],
        "mujoco_runtime": mujoco_runtime,
        "ogbench": packages["ogbench"],
        "pygame": packages["pygame"],
        "pymunk": packages["pymunk"],
        "shapely": packages["shapely"],
        "stable_worldmodel": packages["stable-worldmodel"],
        "task_environment_source": _environment_source(task),
    }
    numerics = {
        "numpy": packages["numpy"],
        "python": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "python_implementation": platform.python_implementation(),
        "task": task,
        "scipy": packages["scipy"],
        "torch": packages["torch"],
    }
    execution = stable_worldmodel_source_audit()
    accelerator = None
    if torch_module is not None:
        numerics["cuda_runtime"] = getattr(torch_module.version, "cuda", None)
        numerics["cudnn"] = (
            torch_module.backends.cudnn.version()
            if torch_module.backends.cudnn.is_available()
            else None
        )
        if torch_module.cuda.is_available():
            properties = torch_module.cuda.get_device_properties(
                torch_module.cuda.current_device()
            )
            accelerator = {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "driver_version": _nvidia_driver_version(),
            }
            numerics["accelerator"] = accelerator
        numerics["torch_controls"] = _torch_numerical_controls(torch_module)

    clear_sources = _clear_lewm_source_audit()

    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "task": task,
        "packages": packages,
        "physics": physics,
        "numerics": numerics,
        "execution": execution,
        "clear_lewm_execution_sources": clear_sources,
        "accelerator": accelerator,
        "rendering": {
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        },
    }
    record["physics_fingerprint"] = _fingerprint(physics)
    record["numerics_fingerprint"] = _fingerprint(numerics)
    record["execution_fingerprint"] = _fingerprint(
        {
            "stable_worldmodel": execution["sources"],
            "clear_lewm": clear_sources,
        }
    )
    return record
