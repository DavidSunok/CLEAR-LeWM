from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

LEGACY_RUNTIME_MODULES = frozenset({"jepa", "module"})


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _set_path_priority(path: Path, *, first: bool) -> None:
    value = str(path.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != value]
    if first:
        sys.path.insert(0, value)
    else:
        sys.path.append(value)


def configure_import_paths(
    upstream_dir: str | Path,
    runtime_dir: str | Path | None = None,
) -> dict:
    """Configure explicit runtime precedence without letting upstream shadow it."""
    upstream = Path(upstream_dir).resolve()
    if not upstream.is_dir():
        raise FileNotFoundError(f"LeWM upstream directory does not exist: {upstream}")

    runtime = Path(runtime_dir).resolve() if runtime_dir is not None else None
    if runtime is not None:
        if not runtime.is_dir():
            raise FileNotFoundError(f"Runtime directory does not exist: {runtime}")
        if runtime == upstream:
            raise ValueError(
                "runtime_dir and upstream_dir must be different directories"
            )
        _set_path_priority(runtime, first=True)

    # The upstream root is needed for source-tree installs of stable-worldmodel,
    # but legacy top-level files such as module.py must remain fallback imports.
    _set_path_priority(upstream, first=False)
    importlib.invalidate_caches()
    return {
        "custom_runtime": runtime is not None,
        "runtime_dir": runtime,
        "upstream_dir": upstream,
    }


def _collect_targets(value) -> set[str]:
    targets: set[str] = set()
    if isinstance(value, dict):
        target = value.get("_target_")
        if isinstance(target, str):
            targets.add(target)
        for item in value.values():
            targets.update(_collect_targets(item))
    elif isinstance(value, list):
        for item in value:
            targets.update(_collect_targets(item))
    return targets


def _resolve_target(target: str):
    parts = target.split(".")
    last_error: ModuleNotFoundError | None = None
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
                raise
            last_error = exc
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute)
        return obj
    raise ImportError(f"Cannot resolve Hydra target {target!r}") from last_error


def _source_file(obj) -> Path | None:
    try:
        source = inspect.getsourcefile(obj) or inspect.getfile(obj)
    except (OSError, TypeError):
        return None
    return Path(source).resolve()


def _source_record(
    source: Path | None,
    upstream_dir: Path,
    runtime_dir: Path | None,
) -> dict:
    if source is None:
        return {"scope": "unknown", "file": None, "sha256": None}
    if runtime_dir is not None and _is_within(source, runtime_dir):
        scope = "runtime"
        display = source.relative_to(runtime_dir).as_posix()
    elif _is_within(source, upstream_dir):
        scope = "upstream"
        display = source.relative_to(upstream_dir).as_posix()
    else:
        scope = "installed"
        display = source.name
    digest = None
    if source.is_file():
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {"scope": scope, "file": display, "sha256": digest}


def audit_hydra_targets(
    config_path: str | Path,
    upstream_dir: str | Path,
    runtime_dir: str | Path | None = None,
) -> dict:
    """Resolve every Hydra target and verify legacy targets stay in runtime_dir."""
    config_path = Path(config_path).resolve()
    upstream = Path(upstream_dir).resolve()
    runtime = Path(runtime_dir).resolve() if runtime_dir is not None else None
    config = json.loads(config_path.read_text())
    records = []
    for target in sorted(_collect_targets(config)):
        try:
            obj = _resolve_target(target)
        except (AttributeError, ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"Cannot resolve Hydra target {target!r} from {config_path.name}"
            ) from exc
        source = _source_file(obj)
        root_module = target.partition(".")[0]
        if runtime is not None and root_module in LEGACY_RUNTIME_MODULES:
            if source is None or not _is_within(source, runtime):
                actual = str(source) if source is not None else "unknown"
                raise RuntimeError(
                    f"Hydra target {target!r} resolved outside the requested runtime: "
                    f"{actual} (expected under {runtime})"
                )
        records.append(
            {
                "target": target,
                "resolved_object": (
                    f"{getattr(obj, '__module__', type(obj).__module__)}."
                    f"{getattr(obj, '__qualname__', type(obj).__qualname__)}"
                ),
                "source": _source_record(source, upstream, runtime),
            }
        )
    return {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "custom_runtime_verified": runtime is not None,
        "targets": records,
    }
