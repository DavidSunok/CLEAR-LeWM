#!/usr/bin/env python3
"""Download, verify, and convert the four official LeWM HF checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

TASKS = {
    "pusht": (
        "lewm-pusht",
        "22b330c28c27ead4bfd1888615af1340e3fe9052",
        "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
    ),
    "cube": (
        "lewm-cube",
        "b0747c5002e86d2ce8f3cd8178004b97524c587d",
        "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89",
    ),
    "tworoom": (
        "lewm-tworooms",
        "77adaae0bc31deab21c93740d1f8bb947cd0bdec",
        "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd",
    ),
    "reacher": (
        "lewm-reacher",
        "62adae4b71dc474ddf8f794c476ebfe737a743ca",
        "eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def _legacy_encoder_key(key: str) -> str:
    prefix = "encoder.encoder.layer."
    if not key.startswith(prefix):
        return key
    key = key.replace(prefix, "encoder.layers.", 1)
    replacements = {
        ".attention.attention.query.": ".attention.q_proj.",
        ".attention.attention.key.": ".attention.k_proj.",
        ".attention.attention.value.": ".attention.v_proj.",
        ".attention.output.dense.": ".attention.o_proj.",
        ".intermediate.dense.": ".mlp.fc1.",
        ".output.dense.": ".mlp.fc2.",
    }
    for source, target in replacements.items():
        key = key.replace(source, target)
    return key


def _build_model(config: dict, upstream_dir: Path):
    sys.path.insert(0, str(upstream_dir.resolve()))
    import stable_pretraining as spt
    import torch
    from jepa import JEPA
    from module import MLP, ARPredictor, Embedder

    encoder_cfg = config["encoder"]
    encoder = spt.backbone.utils.vit_hf(
        encoder_cfg["size"],
        patch_size=encoder_cfg["patch_size"],
        image_size=encoder_cfg["image_size"],
        pretrained=False,
        use_mask_token=False,
    )
    predictor_cfg = {
        key: value
        for key, value in config["predictor"].items()
        if not key.startswith("_")
    }
    action_cfg = {
        key: value
        for key, value in config["action_encoder"].items()
        if not key.startswith("_")
    }

    def mlp(name: str):
        cfg = config[name]
        return MLP(
            input_dim=cfg["input_dim"],
            output_dim=cfg["output_dim"],
            hidden_dim=cfg["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(**predictor_cfg),
        action_encoder=Embedder(**action_cfg),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "le-wm",
    )
    args = parser.parse_args()

    import torch

    source_root = args.cache_dir / "official-hf"
    output_root = args.cache_dir / "official"
    for task, (repo, revision, expected_sha) in TASKS.items():
        source = source_root / task
        for filename in ("config.json", "weights.pt"):
            url = (
                f"https://huggingface.co/quentinll/{repo}/resolve/{revision}/{filename}"
            )
            _download(url, source / filename)
        actual_sha = _sha256(source / "weights.pt")
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"{task}: weights SHA-256 {actual_sha} != {expected_sha}"
            )

        config = json.loads((source / "config.json").read_text())
        model = _build_model(config, args.upstream_dir)
        state = torch.load(
            source / "weights.pt", map_location="cpu", weights_only=False
        )
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            remapped = {_legacy_encoder_key(key): value for key, value in state.items()}
            model.load_state_dict(remapped, strict=True)

        checkpoint_dir = args.cache_dir / "checkpoints" / "official" / task
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "weights.pt")
        shutil.copy2(source / "config.json", checkpoint_dir / "config.json")
        (checkpoint_dir / "source.json").write_text(
            json.dumps(
                {
                    "repository": f"quentinll/{repo}",
                    "revision": revision,
                    "source_weights_sha256": actual_sha,
                    "runtime_weights_sha256": _sha256(checkpoint_dir / "weights.pt"),
                    "tensors_loaded": len(model.state_dict()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        output = output_root / task / "lewm_object.ckpt"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, output)
        print(f"{task}: official/{task} ({actual_sha}); legacy object: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
