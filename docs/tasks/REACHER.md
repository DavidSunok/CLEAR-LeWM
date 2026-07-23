# Reacher evaluation guide

> **v0.5 Moderate contract:** first-hit joint matching under the actual DMC
> joint topology: periodic shoulder, bounded wrist.

[Back to the task contracts](../../README.md#what-v05-fixes) | [Normative
v0.5 specification](../../EVALUATION_SPEC.md)

## Joint topology

The shoulder is an unbounded hinge, so equivalent angles across `-pi/pi` must
use the shortest periodic error:

```text
e_shoulder = abs(atan2(sin(q - q_goal), cos(q - q_goal))).
```

The wrist is limited to its physical range. It therefore uses
`e_wrist = abs(q - q_goal)`; wrapping both joints would incorrectly make two
distant wrist states appear close.

## Gates

| Gate | v0.5 Moderate definition |
|---|---|
| Pair source | same episode, exact `+25` step future |
| Sampling | episode-balanced |
| Initial success | remove `max(e_shoulder, e_wrist) < 0.05 rad` |
| Runtime success | `max(e_shoulder, e_wrist) < 0.05 rad` |
| Temporal rule | first hit; no hold |
| Extra difficulty | none |

The policy controls DMC from the recorded start. Joint state is used only by
the external evaluator after each step. Holding and terminal speed may be
reported separately but do not alter Moderate SR.

Official LeWM reaches **40%** on the fixed v0.5 Moderate manifest; paired
random reaches 5%. This number is intentionally not compared as if it were the
old all-joints-wrapped v0.3 predicate.

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.5/reacher/moderate-seed42-n100.json \
  --policy official/reacher/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/reacher_random.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/v0.5/reacher-moderate-random-seed42-n100.json \
  --output results/reacher-v05-moderate.json
```
