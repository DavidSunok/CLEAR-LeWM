# Cube evaluation guide

> **Moderate follows OGBench's cube-position task. Strict requires precise cube
> position and symmetry-aware orientation. Neither mode scores robot pose.**

[Back to the task contracts](../../README.md#two-auditable-modes) | [Normative
v0.5 specification](../../EVALUATION_SPEC.md)

## Why the cube is the target

Cube is an OGBench goal-conditioned manipulation task. The robot and gripper
are tools; their terminal configuration is not part of task completion. The
external evaluator reads the cube pose after physical MuJoCo steps while the
policy receives only its declared observations and goal image.

## Gates

| Gate | v0.5 Moderate | v0.5 Strict |
|---|---|---|
| Cube center | distance `<= 0.04 m` | distance `<= 0.03 m` |
| Cube orientation | not scored | `<= 15 deg` over 24 equivalent cube rotations |
| Robot/gripper pose | not scored | not scored |
| Temporal rule | first hit | hold 3 steps |

Moderate minimally preserves the released OGBench success predicate after
removing initially solved pairs. Strict evaluates fine object placement. Raw
quaternion distance is not used because it would separate physically
equivalent cube rotations.

## Official reference

- Moderate seed 42: official LeWM **51%**, paired random **15%**.
- Strict seeds 0/1/42: official LeWM **28/26/25%**; random **3/7/8%**.

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.5/cube/strict-seed42-n100.json \
  --policy official/cube/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/cube_single_expert.h5 \
  --num-samples 300 --n-steps 30 --topk 30 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/v0.5/cube-strict-random-seed42-n100.json \
  --output results/cube-v05-strict.json
```
