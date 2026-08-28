# Reacher evaluation guide

> **Moderate fixes the released joint-topology error. Strict evaluates the
> physical fingertip endpoint under a tighter, persistent gate.**

[Back to the task contracts](../../README.md#two-auditable-modes) | [v0.8
normative specification](../../EVALUATION_SPEC_V08.md) | [v0.5 historical
specification](../../EVALUATION_SPEC.md)

> **v0.8 runtime correction:** when CLEAR owns the Reacher success predicate,
> it also disables the underlying dm-control task termination. This prevents
> an inner `LAST -> auto-reset` transition before the outer action is scored.
> The gate is restored after both `reset()` and `compile_model()`.

## Moderate: preserve the joint target

The shoulder is an unbounded hinge and uses shortest periodic error:

```text
e_shoulder = abs(atan2(sin(q-q_goal), cos(q-q_goal)))
```

The wrist is physically bounded and uses `abs(q-q_goal)`. Wrapping both joints
can incorrectly make distant wrist states appear close. Moderate succeeds on
the first step where `max(e_shoulder, e_wrist) < 0.05 rad`.

## Strict: score the endpoint

Multiple joint configurations can realize the same task-relevant fingertip
location. Strict therefore computes the physical 2-D fingertip position
induced by the goal configuration, restores the rollout state, and evaluates
the actual fingertip after every policy step. Success requires endpoint
distance `<= 0.01 m` for 2 consecutive steps.

| Gate | v0.8 Moderate | v0.8 Strict |
|---|---|---|
| Target | goal joint configuration | physical fingertip endpoint |
| Threshold | max corrected joint error `< 0.05 rad` | endpoint distance `<= 0.01 m` |
| Temporal rule | first hit | hold 2 steps |

## v0.5 historical reference

- Moderate seeds 0/1/42: official LeWM **51/47/40%**; random **5/3/5%**.
- Strict seeds 0/1/42: official LeWM **41/37/51%**; random **1/7/7%**.

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.8/reacher/strict-seed42-n100.json \
  --policy official/reacher/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/reacher.h5 \
  --num-samples 300 --n-steps 30 --topk 30 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/v0.8/runs/random/reacher-strict-seed42.json \
  --output results/v0.8/runs/official-lewm/reacher-strict-seed42.json
```
