# Official LeWM checkpoint record

Audit date: 2026-07-22. Machine-readable source, config, and runtime hashes are
frozen in [`checkpoints/official-v0.3.json`](../checkpoints/official-v0.3.json).

The four model mirrors linked by the upstream LeWM repository are pinned below.
`scripts/prepare_official_checkpoints.py` downloads these revisions, verifies
`weights.pt`, constructs the upstream JEPA architecture, strictly loads all 303
tensors, and writes both current runner and legacy object checkpoints.

| Task | Hugging Face repository revision | `weights.pt` SHA-256 |
|---|---|---|
| PushT | `22b330c28c27ead4bfd1888615af1340e3fe9052` | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| Cube | `b0747c5002e86d2ce8f3cd8178004b97524c587d` | `2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89` |
| TwoRoom | `77adaae0bc31deab21c93740d1f8bb947cd0bdec` | `566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd` |
| Reacher | `62adae4b71dc474ddf8f794c476ebfe737a743ca` | `eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469` |

Previously cached PushT and Cube object checkpoints were compared tensor by
tensor against these mirrors. After the deterministic ViT key rename between
legacy and current `stable-pretraining` layouts, all 303 tensors matched
exactly for both tasks. Reacher and TwoRoom were reconstructed directly from
the pinned mirrors.

Expected evaluator policy names after conversion are:

```text
official/pusht
official/cube
official/reacher
official/tworoom
```

Object checkpoint serialization hashes can vary with Python and PyTorch even
when every tensor is identical. The immutable source weight hash above is the
checkpoint identity used by CLEAR-LeWM.

The runnable `weights.pt` is re-serialized from the strictly loaded model so
its ViT keys match the installed `stable-pretraining` layout. This avoids a
dangerous failure mode where `strict=False` accepts all encoder keys as missing
and leaves a randomly initialized backbone. Runtime weights are accepted only
after a 303/303 tensor load; the pinned source hash remains the identity.
