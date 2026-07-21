from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
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


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _fingerprint(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _environment_source(task: str | None) -> dict | None:
    module_name = TASK_ENVIRONMENT_MODULES.get(task or "")
    if module_name is None:
        return None
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
        "task": task,
        "scipy": packages["scipy"],
        "torch": packages["torch"],
    }
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
            }
            numerics["accelerator"] = accelerator

    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "task": task,
        "packages": packages,
        "physics": physics,
        "numerics": numerics,
        "accelerator": accelerator,
        "rendering": {
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        },
    }
    record["physics_fingerprint"] = _fingerprint(physics)
    record["numerics_fingerprint"] = _fingerprint(numerics)
    return record
