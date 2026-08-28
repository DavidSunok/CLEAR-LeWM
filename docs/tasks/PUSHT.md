# PushT evaluation guide

> **Moderate preserves LeWM's full goal-state predicate. Strict scores only the
> T block under a tighter pose gate.**

[Back to the task contracts](../../README.md#two-auditable-modes) | [Normative
v0.5 specification](../../EVALUATION_SPEC.md)

## Task and evaluation flow

The policy receives a future RGB goal and controls the Pymunk environment from
a recorded start. CLEAR-LeWM never replays expert actions. After each policy
step, the external evaluator reads pusher and T-block state only to decide
success.

1. Select `goal = start + 25` from the same episode.
2. Remove the pair if its start already satisfies the selected mode.
3. Reset to the start and provide the future RGB goal.
4. Execute the evaluated policy for at most 50 steps.
5. Apply the selected gate below.

## Gates

| Gate | v0.5 Moderate | v0.5 Strict |
|---|---|---|
| Position state | pusher + T block, `||state[:4]-goal[:4]|| < 20 px` | T block only, `< 10 px` |
| T orientation | shortest wrapped error `< 20 deg` | shortest wrapped error `< 10 deg` |
| Temporal rule | first hit | hold 3 steps |
| Extra displacement | none | none |

Moderate stays aligned with the released five-dimensional state predicate and
with LeWM's full-goal-image latent-MSE planner. Strict deliberately changes the
success semantics: once the T is correctly placed, the final pusher position
is task-irrelevant.

This distinction matters because a full-image latent cost may still penalize a
successful Strict state when the pusher differs from the goal image. CLEAR-LeWM
is the external completion evaluator; it does not silently change a method's
planning cost. Authors should disclose whether their cost is full-image,
object-masked, or task-aware.

## Official reference

- Moderate seeds 0/1/42: official LeWM **84/87/88%**; random **4/5/3%**.
- Strict seeds 0/1/42: official LeWM **66/74/71%**; random **4/4/7%**.

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.5/pusht/strict-seed42-n100.json \
  --policy official/pusht/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/pusht_expert_train.h5 \
  --num-samples 300 --n-steps 30 --topk 30 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/v0.5/pusht-strict-random-seed42-n100.json \
  --output results/pusht-v05-strict.json
```
