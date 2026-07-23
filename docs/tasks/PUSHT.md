# PushT evaluation guide

> **v0.5 Moderate contract:** reproduce the released full goal-state pose:
> pusher position, T-block position, and T-block orientation.

[Back to the task contracts](../../README.md#what-v05-fixes) | [Normative
v0.5 specification](../../EVALUATION_SPEC.md)

## Evaluation flow

1. Select `goal = start + 25` from the same expert episode.
2. Remove the pair if its start already satisfies the rollout predicate.
3. Reset Pymunk to the start and provide the future RGB goal image.
4. Execute only the evaluated policy's actions for at most 50 steps.
5. Declare first-hit success when both position and angle gates pass.

## Gates

| Gate | v0.5 Moderate definition |
|---|---|
| Pair source | same episode, exact `+25` step future |
| Sampling | episode-balanced, one canonical pair per selected episode |
| Initial success | removed with the same full-state predicate |
| Position | `||state[:4] - goal[:4]||_2 < 20` |
| T angle | shortest wrapped difference `< pi/9` (`20 deg`) |
| Temporal rule | first hit; no hold |
| Extra difficulty | none |

**Important:** v0.5 does not use the v0.3 block-only reinterpretation. PushT's
released goal image contains both the pusher and T block, and the upstream CEM
cost attempts to match that complete image. Keeping the released predicate
makes the benchmark and planning target semantically aligned.

The evaluator still records block-only displacement and angle diagnostics.
They are analysis variables, not alternate success gates.

Official LeWM reaches **88%** on the fixed v0.5 Moderate manifest; paired
random reaches 3%.

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.5/pusht/moderate-seed42-n100.json \
  --policy official/pusht/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/pusht_expert_train.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/v0.5/pusht-moderate-random-seed42-n100.json \
  --output results/pusht-v05-moderate.json
```
