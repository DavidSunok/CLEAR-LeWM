# Cube evaluation guide

## What the task should measure

Cube asks the robot to place a physical cube at a visual target pose. The
dataset explicitly samples target yaw, so position alone is incomplete.

## Released predicate and failure mode

The upstream success function checks only position within `0.04 m`, ignoring
the sampled goal orientation. A naive repair based on raw quaternion distance
creates the opposite problem: it rejects the 24 proper rotations that describe
the same physical orientation of an unmarked cube.

## CLEAR v0.3 correction

CLEAR minimizes geodesic rotation error over the cube's 24-element rotational
symmetry group and combines it with position.

| Mode | Position | Symmetry-aware angle | Hold |
|---|---:|---:|---:|
| Moderate | `<=0.04 m` | `<=30 deg` | 3 steps |
| Strict | `<=0.03 m` | `<=15 deg` | 5 steps |

A second settled-placement metric may additionally report linear speed below
`0.05 m/s` and angular speed below `0.5 rad/s`. It must be labeled separately.

## Audited result

| Mode | LeWM | Random | Excess over random |
|---|---:|---:|---:|
| Moderate | **43%** | 4% | +39 pp |
| Strict | **18%** | 3% | +15 pp |

The symmetry-aware Moderate predicate recovers seven valid placements rejected
by raw quaternion matching on the same fixed pairs.

![Cube symmetry-aware trace](../../assets/task_gifs/cube.gif)

## Run it

```bash
clear-lewm evaluate \
  --manifest manifests/v0.3/cube/strict-seed42-n100.json \
  --policy official/cube --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/cube_single_expert.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --output results/cube-strict.json
```

