from __future__ import annotations

import hashlib
import json
import os
import random
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MethodType

import numpy as np

from .datasets import file_sha256, metadata_fingerprint
from .environment import collect_environment, require_official_stable_worldmodel
from .manifests import load_manifest
from .metrics import load_success_trace, summarize_success
from .protocols import ProtocolSpec, normalize_task, protocol_from_dict
from .runtime import audit_hydra_targets, configure_import_paths
from .tasks import quaternion_angle_deg

OFFICIAL_DATASETS = {
    "pusht": "pusht_expert_train",
    "reacher": "dmc/reacher_random",
    "tworoom": "tworoom",
    "cube": "ogbench/cube_single_expert",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    return value


def _seed_everything(seed: int, cpu_threads: int | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        effective_threads = (
            cpu_threads
            if cpu_threads is not None
            else int(os.environ.get("CLEAR_LEWM_CPU_THREADS", "1"))
        )
        if effective_threads < 1:
            raise ValueError("cpu_threads must be at least 1")
        torch.set_num_threads(effective_threads)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _portable_manifest_path(path: Path) -> str:
    parts = path.parts
    if "manifests" in parts:
        index = parts.index("manifests")
        return Path(*parts[index:]).as_posix()
    return path.name


def _checkpoint_file(policy: str, data_root: Path) -> tuple[Path | None, Path]:
    candidate = data_root / "checkpoints" / policy
    if candidate.is_file():
        return candidate, candidate.parent
    if candidate.is_dir():
        files = sorted(candidate.glob("*.pt"))
        return (files[0] if len(files) == 1 else None), candidate
    return None, candidate


def _checkpoint_record(policy: str, data_root: Path) -> dict | None:
    if policy == "random":
        return None
    weights, directory = _checkpoint_file(policy, data_root)
    record = {"policy_id": policy}
    if weights is not None:
        record["runtime_file"] = weights.name
        record["runtime_sha256"] = file_sha256(weights)
    config = directory / "config.json"
    if config.exists():
        record["config_sha256"] = file_sha256(config)
    source = directory / "source.json"
    if source.exists():
        record["source"] = json.loads(source.read_text())
    return record


def _audit_checkpoint_targets(
    policy: str,
    data_root: Path,
    upstream_dir: Path,
    runtime_dir: Path | None,
) -> dict:
    _, directory = _checkpoint_file(policy, data_root)
    config = directory / "config.json"
    if not config.is_file():
        return {"available": False, "custom_runtime_verified": False, "targets": []}
    audit = audit_hydra_targets(config, upstream_dir, runtime_dir)
    audit["available"] = True
    return audit


def _audit_checkpoint_state(model, policy: str, data_root: Path, strict: bool) -> dict:
    import torch

    checkpoint, _ = _checkpoint_file(policy, data_root)
    if checkpoint is None:
        if strict:
            raise RuntimeError(f"Cannot audit an ambiguous checkpoint: {policy}")
        return {"available": False, "strict_required": strict}
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    audit = {
        "available": True,
        "strict_required": strict,
        "checkpoint_tensors": len(state),
        "model_tensors": len(model.state_dict()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    if strict and (audit["missing_keys"] or audit["unexpected_keys"]):
        raise RuntimeError(
            "Strict checkpoint audit failed: "
            f"missing={audit['missing_keys']}, unexpected={audit['unexpected_keys']}"
        )
    return audit


def _load_paired_random_trace(
    path: str | Path,
    *,
    manifest_sha256: str,
    task: str,
    protocol_name: str,
    policy_seed: int,
):
    path = Path(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Paired random results must be a CLEAR result JSON object")
    expected = {
        "schema_version": "clear-lewm-result-v1",
        "manifest_sha256": manifest_sha256,
        "task": task,
        "policy_seed": policy_seed,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if payload.get("protocol", {}).get("name") != protocol_name:
        mismatches.append("protocol")
    if payload.get("checkpoint") is not None:
        mismatches.append("checkpoint")
    if mismatches:
        raise ValueError(
            "Paired random result identity mismatch: " + ", ".join(mismatches)
        )
    return load_success_trace(path)


def _install_batched_lewm_criterion(model) -> None:
    """Fix the missing CEM sample axis in canonical LeWM's batched cost."""
    import torch.nn.functional as functional

    def criterion(self, info_dict: dict):
        predicted = info_dict["predicted_emb"]
        goal = info_dict["goal_emb"]
        if goal.ndim == predicted.ndim - 1:
            goal = goal.unsqueeze(1)
        goal = goal[..., -1:, :].expand_as(predicted)
        return functional.mse_loss(
            predicted[..., -1:, :],
            goal[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, predicted.ndim)))

    model.criterion = MethodType(criterion, model)


def _image_transform(image_size: int):
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            transforms.Resize(size=image_size),
        ]
    )


def _compose_config(task: str, upstream_dir: Path):
    from hydra import compose, initialize_config_dir

    config_dir = upstream_dir / "config" / "eval"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        return compose(config_name=task)


def _install_cube_success(
    world,
    position_threshold_m: float,
    orientation_threshold_deg: float,
    sustained_steps: int,
) -> None:
    def patch_environment(env) -> None:
        original_post_step = env.post_step
        original_set_target = env.set_target_pos
        env._clear_lewm_hold_count = 0

        def set_target_pos(self, cube_id, target_pos, target_quat=None):
            result = original_set_target(cube_id, target_pos, target_quat)
            self._clear_lewm_hold_count = 0
            self._success = False
            return result

        def post_step(self):
            original_post_step()
            qpos = np.asarray(self._data.joint("object_joint_0").qpos)
            target_id = self._cube_target_mocap_ids[0]
            target_pos = np.asarray(self._data.mocap_pos[target_id])
            target_quat = np.asarray(self._data.mocap_quat[target_id])
            position_ok = np.linalg.norm(qpos[:3] - target_pos) <= position_threshold_m
            angle_deg = float(
                quaternion_angle_deg(qpos[None, 3:7], target_quat[None])[0]
            )
            pose_ok = bool(position_ok and angle_deg <= orientation_threshold_deg)
            self._clear_lewm_hold_count = (
                self._clear_lewm_hold_count + 1 if pose_ok else 0
            )
            self._success = self._clear_lewm_hold_count >= sustained_steps

        env.set_target_pos = MethodType(set_target_pos, env)
        env.post_step = MethodType(post_step, env)

    for wrapped in world.envs.envs:
        patch_environment(wrapped.unwrapped)


def _install_pusht_success(world, protocol: ProtocolSpec) -> None:
    def patch_environment(env) -> None:
        original_step = env.step
        original_set_goal = env._set_goal_state
        env._clear_lewm_hold_count = 0

        def set_goal_state(self, goal_state):
            result = original_set_goal(goal_state)
            self._clear_lewm_hold_count = 0
            return result

        def step(self, action):
            observation, reward, _, truncated, info = original_step(action)
            state = np.asarray(observation["state"])
            goal = np.asarray(self.goal_state)
            position_error = float(np.linalg.norm(goal[:4] - state[:4]))
            angle_error = abs(float(goal[4] - state[4]))
            angle_error = min(angle_error, 2.0 * np.pi - angle_error)
            success = (
                position_error < protocol.pusht_position_threshold
                and np.degrees(angle_error) < protocol.pusht_angle_threshold_deg
            )
            self._clear_lewm_hold_count = (
                self._clear_lewm_hold_count + 1 if success else 0
            )
            terminated = self._clear_lewm_hold_count >= protocol.hold_steps("pusht")
            info["clear_lewm_hold_count"] = self._clear_lewm_hold_count
            return observation, reward, terminated, truncated, info

        env._set_goal_state = MethodType(set_goal_state, env)
        env.step = MethodType(step, env)

    for wrapped in world.envs.envs:
        patch_environment(wrapped.unwrapped)


def _install_tworoom_success(world, protocol: ProtocolSpec) -> None:
    def patch_environment(env) -> None:
        original_step = env.step
        original_set_goal = env._set_goal_state
        env._clear_lewm_hold_count = 0

        def set_goal_state(self, goal_state):
            result = original_set_goal(goal_state)
            self._clear_lewm_hold_count = 0
            return result

        def step(self, action):
            observation, reward, _, truncated, info = original_step(action)
            distance = float(
                np.linalg.norm(
                    np.asarray(self.agent_position) - np.asarray(self.target_position)
                )
            )
            success = distance < protocol.tworoom_distance_threshold
            self._clear_lewm_hold_count = (
                self._clear_lewm_hold_count + 1 if success else 0
            )
            terminated = self._clear_lewm_hold_count >= protocol.hold_steps("tworoom")
            info["clear_lewm_hold_count"] = self._clear_lewm_hold_count
            return observation, reward, terminated, truncated, info

        env._set_goal_state = MethodType(set_goal_state, env)
        env.step = MethodType(step, env)

    for wrapped in world.envs.envs:
        patch_environment(wrapped.unwrapped)


def _install_reacher_success(world, protocol: ProtocolSpec) -> None:
    def patch_environment(env) -> None:
        original_step = env.step
        original_set_target = env.set_target_qpos
        env._clear_lewm_hold_count = 0

        def suppress_upstream_termination(self, step):
            return False

        def set_target_qpos(self, target_qpos):
            result = original_set_target(target_qpos)
            self._clear_lewm_hold_count = 0
            return result

        def step(self, action):
            observation, reward, _, truncated, info = original_step(action)
            qpos = np.asarray(self.env.physics.data.qpos)
            target = np.asarray(self.env.task.target_qpos)
            joint_error = float(np.max(np.abs(qpos - target)))
            success = joint_error < protocol.reacher_joint_threshold_rad
            self._clear_lewm_hold_count = (
                self._clear_lewm_hold_count + 1 if success else 0
            )
            terminated = self._clear_lewm_hold_count >= protocol.hold_steps("reacher")
            info["clear_lewm_hold_count"] = self._clear_lewm_hold_count
            return observation, reward, terminated, truncated, info

        env._is_terminated = MethodType(suppress_upstream_termination, env)
        env.set_target_qpos = MethodType(set_target_qpos, env)
        env.step = MethodType(step, env)

    for wrapped in world.envs.envs:
        patch_environment(wrapped.unwrapped)


def _install_task_success(world, task: str, protocol: ProtocolSpec) -> None:
    if task == "cube":
        assert protocol.cube_orientation_threshold_deg is not None
        _install_cube_success(
            world,
            position_threshold_m=protocol.cube_position_threshold_m,
            orientation_threshold_deg=protocol.cube_orientation_threshold_deg,
            sustained_steps=protocol.hold_steps("cube"),
        )
    elif task == "pusht":
        _install_pusht_success(world, protocol)
    elif task == "reacher":
        _install_reacher_success(world, protocol)
    else:
        _install_tworoom_success(world, protocol)


def evaluate_manifest(
    manifest_path: str | Path,
    policy: str,
    output: str | Path,
    cache_dir: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_path: str | Path | None = None,
    upstream_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    policy_seed: int | None = None,
    num_samples: int | None = None,
    n_steps: int | None = None,
    topk: int | None = None,
    random_results: str | Path | None = None,
    video_dir: str | Path | None = None,
    policy_label: str | None = None,
    solver_batch_size: int | None = None,
    cpu_threads: int | None = None,
    matmul_precision: str | None = None,
    strict_checkpoint: bool = False,
    allow_modified_stable_worldmodel: bool = False,
) -> dict:
    run_started = time.perf_counter()
    os.environ.setdefault("MUJOCO_GL", "egl")
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    task = normalize_task(manifest["task"])
    protocol = protocol_from_dict(manifest["protocol"])
    upstream_dir = Path(
        upstream_dir or Path(__file__).resolve().parents[1] / "third_party" / "le-wm"
    ).resolve()
    runtime_dir = Path(runtime_dir).resolve() if runtime_dir is not None else None
    import_paths = configure_import_paths(upstream_dir, runtime_dir)

    try:
        import hydra
        import stable_worldmodel as swm
        import torch
        from omegaconf import OmegaConf, open_dict
        from sklearn import preprocessing
    except ImportError as exc:
        raise RuntimeError(
            "LeWM evaluation dependencies are missing. Install with "
            "`pip install -e '.[lewm]'`."
        ) from exc

    if not allow_modified_stable_worldmodel:
        require_official_stable_worldmodel()

    seed = int(policy_seed if policy_seed is not None else manifest["policy_seed"])
    random_trace = (
        _load_paired_random_trace(
            random_results,
            manifest_sha256=manifest_sha256,
            task=task,
            protocol_name=protocol.name,
            policy_seed=seed,
        )
        if random_results
        else None
    )
    _seed_everything(seed, cpu_threads=cpu_threads)
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    cfg = _compose_config(task, upstream_dir)
    with open_dict(cfg):
        cfg.eval.num_eval = len(manifest["pairs"])
        cfg.eval.goal_offset_steps = int(protocol.goal_offset)
        cfg.eval.eval_budget = int(protocol.eval_budget)
        cfg.world.num_envs = len(manifest["pairs"])
        cfg.world.max_episode_steps = 2 * int(protocol.eval_budget)
        cfg.seed = seed
        cfg.policy = policy
        if cache_dir is not None:
            cfg.cache_dir = str(Path(cache_dir).resolve())
        if dataset_name is not None:
            cfg.eval.dataset_name = dataset_name
        if num_samples is not None:
            cfg.solver.num_samples = int(num_samples)
        if n_steps is not None:
            cfg.solver.n_steps = int(n_steps)
        if topk is not None:
            cfg.solver.topk = int(topk)
        if solver_batch_size is not None:
            if solver_batch_size < 1:
                raise ValueError("solver_batch_size must be at least 1")
            cfg.solver.batch_size = int(solver_batch_size)

    transform = {
        "pixels": _image_transform(cfg.eval.img_size),
        "goal": _image_transform(cfg.eval.img_size),
    }
    data_root = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    resolved_dataset_name = dataset_name or OFFICIAL_DATASETS[task]
    dataset_kwargs = {
        "keys_to_cache": cfg.dataset.keys_to_cache,
        "cache_dir": data_root,
    }
    if dataset_path is not None:
        dataset_kwargs["path"] = Path(dataset_path).resolve()
        dataset = swm.data.HDF5Dataset(**dataset_kwargs)
    else:
        dataset = swm.data.HDF5Dataset(resolved_dataset_name, **dataset_kwargs)

    expected_fingerprint = manifest["dataset"]["fingerprint"]
    dataset_file = Path(dataset.h5_path)
    if expected_fingerprint["kind"] == "file-sha256":
        actual_fingerprint = file_sha256(dataset_file)
    else:
        actual_fingerprint = metadata_fingerprint(dataset_file)
    if actual_fingerprint != expected_fingerprint["value"]:
        raise ValueError(
            "Evaluation dataset does not match the manifest fingerprint: "
            f"{actual_fingerprint} != {expected_fingerprint['value']}"
        )

    process = {}
    for column in cfg.dataset.keys_to_cache:
        if column == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        values = dataset.get_col_data(column)
        values = values[~np.isnan(values).any(axis=1)]
        scaler.fit(values)
        process[column] = scaler
        if column != "action":
            process[f"goal_{column}"] = scaler

    if policy == "random":
        attached_policy = swm.policy.RandomPolicy(seed=seed)
        checkpoint = None
        batched_criterion_patch = False
    else:
        target_audit = _audit_checkpoint_targets(
            policy, data_root, upstream_dir, runtime_dir
        )
        try:
            model = swm.wm.utils.load_pretrained(policy, cache_dir=data_root)
        except TypeError:
            model = swm.wm.utils.load_pretrained(policy)
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        batched_criterion_patch = False
        canonical_lewm = (
            type(model).__module__ == "stable_worldmodel.wm.lewm.lewm"
            and type(model).__name__ == "LeWM"
        )
        if int(cfg.solver.batch_size) > 1 and canonical_lewm:
            _install_batched_lewm_criterion(model)
            batched_criterion_patch = True
        checkpoint = _checkpoint_record(policy, data_root)
        checkpoint["target_audit"] = target_audit
        checkpoint["state_dict_audit"] = _audit_checkpoint_state(
            model, policy, data_root, strict=strict_checkpoint
        )
        plan_config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        attached_policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=plan_config,
            process=process,
            transform=transform,
        )

    episodes = [pair["episode_id"] for pair in manifest["pairs"]]
    start_steps = [pair["start_step"] for pair in manifest["pairs"]]
    world = swm.World(**cfg.world, image_shape=(224, 224))
    try:
        if protocol.success_mode == "task-sustained":
            _install_task_success(world, task, protocol)
        elif task == "cube" and protocol.success_mode == "cube-pose":
            assert protocol.cube_orientation_threshold_deg is not None
            _install_cube_success(
                world,
                position_threshold_m=protocol.cube_position_threshold_m,
                orientation_threshold_deg=protocol.cube_orientation_threshold_deg,
                sustained_steps=protocol.hold_steps("cube"),
            )
        world.set_policy(attached_policy)
        evaluation_started = time.perf_counter()
        raw_metrics = world.evaluate(
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=protocol.goal_offset,
            eval_budget=protocol.eval_budget,
            episodes_idx=episodes,
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=Path(video_dir) if video_dir else None,
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
    finally:
        world.close()

    episode_successes = np.asarray(raw_metrics["episode_successes"], dtype=bool)
    summary = summarize_success(
        episode_successes,
        random_trace=random_trace,
        hold_steps=1,
        seed=seed,
    )
    summary.pop("final_state_success_rate_percent", None)
    summary.pop("sustained_success_rate_percent", None)
    summary.pop("sustained_steps", None)
    result = {
        "schema_version": "clear-lewm-result-v1",
        "task": task,
        "protocol": protocol.to_dict(),
        "policy": policy_label or policy,
        "checkpoint": checkpoint,
        "policy_seed": seed,
        "dataset_name": resolved_dataset_name,
        "dataset_file": (Path(dataset_path).name if dataset_path is not None else None),
        "dataset_fingerprint": expected_fingerprint,
        "manifest": _portable_manifest_path(manifest_path),
        "manifest_sha256": manifest_sha256,
        "criterion": {
            "cube_position_threshold_m": protocol.cube_position_threshold_m,
            "cube_orientation_threshold_deg": protocol.cube_orientation_threshold_deg,
            "pusht_position_threshold": protocol.pusht_position_threshold,
            "pusht_angle_threshold_deg": protocol.pusht_angle_threshold_deg,
            "reacher_joint_threshold_rad": protocol.reacher_joint_threshold_rad,
            "tworoom_distance_threshold": protocol.tworoom_distance_threshold,
            "sustained_steps": protocol.hold_steps(task),
        },
        "solver": {
            "batch_size": OmegaConf.select(cfg, "solver.batch_size"),
            "num_samples": OmegaConf.select(cfg, "solver.num_samples"),
            "n_steps": OmegaConf.select(cfg, "solver.n_steps"),
            "topk": OmegaConf.select(cfg, "solver.topk"),
        },
        "metrics": summary,
        "episode_successes": episode_successes.tolist(),
        "raw_world_metrics": _json_safe(raw_metrics),
        "environment": collect_environment(torch, task=task),
        "versions": {
            "torch": torch.__version__,
            "stable_worldmodel": _package_version("stable-worldmodel"),
        },
        "runtime": {
            "batched_lewm_criterion_patch": batched_criterion_patch,
            "cpu_threads": torch.get_num_threads(),
            "custom_runtime": import_paths["custom_runtime"],
            "evaluation_seconds": evaluation_seconds,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "modified_stable_worldmodel_allowed": (
                allow_modified_stable_worldmodel
            ),
            "total_before_serialization_seconds": time.perf_counter() - run_started,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(_json_safe(result), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return result
