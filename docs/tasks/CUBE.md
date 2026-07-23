# Cube evaluation guide

> **v0.5 Moderate contract:** move the cube center to the target position.
> Cube orientation and the terminal robot/gripper pose are not scored.

[Back to the task contracts](../../README.md#what-v05-fixes) | [Normative
v0.5 specification](../../EVALUATION_SPEC.md)

## Why position only

Cube is the OGBench goal-conditioned manipulation task. Its released success
condition checks object position within 4 cm. v0.5 follows that task definition
instead of adding an orientation requirement. The robot and gripper are tools;
their final pose is never part of success.

## Gates

| Gate | v0.5 Moderate definition |
|---|---|
| Pair source | same episode, exact `+25` step future |
| Sampling | episode-balanced |
| Initial success | remove `||p_cube - p_goal||_2 <= 0.04 m` |
| Runtime success | `||p_cube - p_goal||_2 <= 0.04 m` |
| Orientation | not scored |
| Robot/gripper pose | not scored |
| Temporal rule | first hit; no hold |
| Extra difficulty | none |

The evaluator reads privileged cube position only after each physical MuJoCo
step. The policy receives the visual goal and emits its own actions; expert
actions are not replayed. Orientation and the 24-way symmetry distance remain
available as diagnostics and for archived v0.3 manifests.

Official LeWM reaches **51%** on the fixed v0.5 Moderate manifest; paired
random reaches 15%.

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.5/cube/moderate-seed42-n100.json \
  --policy official/cube/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/cube_single_expert.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/v0.5/cube-moderate-random-seed42-n100.json \
  --output results/cube-v05-moderate.json
```
