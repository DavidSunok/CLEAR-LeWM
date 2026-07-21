from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MethodType

import numpy as np

from .manifests import load_manifest
from .metrics import load_success_trace, summarize_success
from .protocols import get_protocol, normalize_task
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

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


def _install_strict_cube_success(
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


def evaluate_manifest(
    manifest_path: str | Path,
    policy: str,
    output: str | Path,
    cache_dir: str | Path | None = None,
    dataset_name: str | None = None,
    upstream_dir: str | Path | None = None,
    policy_seed: int | None = None,
    num_samples: int | None = None,
    n_steps: int | None = None,
    topk: int | None = None,
    random_results: str | Path | None = None,
    video_dir: str | Path | None = None,
) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    task = normalize_task(manifest["task"])
    protocol = get_protocol(manifest["protocol"]["name"])
    upstream_dir = Path(
        upstream_dir or Path(__file__).resolve().parents[1] / "third_party" / "le-wm"
    ).resolve()
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))

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

    seed = int(policy_seed if policy_seed is not None else manifest["policy_seed"])
    _seed_everything(seed)
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

    world = swm.World(**cfg.world, image_shape=(224, 224))
    if task == "cube" and protocol.cube_orientation_threshold_deg is not None:
        _install_strict_cube_success(
            world,
            position_threshold_m=protocol.cube_position_threshold_m,
            orientation_threshold_deg=protocol.cube_orientation_threshold_deg,
            sustained_steps=protocol.sustained_steps,
        )

    transform = {
        "pixels": _image_transform(cfg.eval.img_size),
        "goal": _image_transform(cfg.eval.img_size),
    }
    data_root = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    resolved_dataset_name = dataset_name or OFFICIAL_DATASETS[task]
    dataset = swm.data.HDF5Dataset(
        resolved_dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=data_root,
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
    else:
        try:
            model = swm.wm.utils.load_pretrained(policy, cache_dir=data_root)
        except TypeError:
            model = swm.wm.utils.load_pretrained(policy)
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
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
    world.set_policy(attached_policy)
    try:
        raw_metrics = world.evaluate(
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=protocol.goal_offset,
            eval_budget=protocol.eval_budget,
            episodes_idx=episodes,
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=Path(video_dir) if video_dir else None,
        )
    finally:
        world.close()

    episode_successes = np.asarray(raw_metrics["episode_successes"], dtype=bool)
    random_trace = load_success_trace(random_results) if random_results else None
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
        "policy": policy,
        "policy_seed": seed,
        "dataset_name": resolved_dataset_name,
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": manifest_sha256,
        "criterion": {
            "cube_position_threshold_m": protocol.cube_position_threshold_m,
            "cube_orientation_threshold_deg": protocol.cube_orientation_threshold_deg,
            "sustained_steps": protocol.sustained_steps,
        },
        "solver": {
            "num_samples": OmegaConf.select(cfg, "solver.num_samples"),
            "n_steps": OmegaConf.select(cfg, "solver.n_steps"),
            "topk": OmegaConf.select(cfg, "solver.topk"),
        },
        "metrics": summary,
        "episode_successes": episode_successes.tolist(),
        "raw_world_metrics": _json_safe(raw_metrics),
        "versions": {
            "torch": torch.__version__,
            "stable_worldmodel": _package_version("stable-worldmodel"),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n")
    return result
