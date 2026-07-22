# Protocol calibration record

Calibration date: 2026-07-22. This document records the v0.3 task-semantic
protocol. All values use official high-epoch LeWM checkpoints, 100 fixed
seed-42 pairs, `300 x 30` CEM, and solver batch size 1.

## Final v0.3 matrix

| Task | Moderate model/random | Strict model/random |
|---|---:|---:|
| PushT | **93% / 7%** | **79% / 2%** |
| Cube | **43% / 4%** | **18% / 3%** |
| Reacher | **90% / 17%** | **36% / 4%** |
| TwoRoom | **61% / 2%** | **24% / 0%** |

Official compatibility remains available for historical comparison. It is not
reinterpreted under v0.3.

## Why calibration changed

### PushT

The v0.2 predicate included final pusher position in both success and pair
difficulty. v0.3 evaluates only T-block pose and uses block translation. The
new manifests contain no block-semantic initial successes.

### Cube

Target yaw is present in the dataset, but raw quaternion matching rejects
physically equivalent cube rotations. v0.3 minimizes orientation error over the
24-element cube rotation group. This changes Moderate from 36% to 43% on the
same pair IDs.

### Reacher

The released task is first-hit. v0.3 preserves that meaning and wraps periodic
joint angles. A separate Strict `0.05 rad, hold 2` audit yielded 1% model and 0%
random, demonstrating that stabilization is a materially different claim.

### TwoRoom

The released endpoint collision can accept paths through a wall or doorframe.
On 100 cross-room Strict goals, original dynamics produced 37% endpoint SR;
only 6% of the unchanged trajectories were route-valid. Swept-disk collision
produced 24% legal SR with 0 invalid routes. Moderate produced 61% / 2%, also
with 0 invalid routes.

## Selection guardrails

Thresholds were accepted only when all of the following held:

1. selected pairs were initially unsolved under the rollout predicate;
2. model and random used the exact same manifest;
3. the official checkpoint remained measurably above random;
4. task-equivalent states were not separated by representation artifacts;
5. physics or topology constraints could not be bypassed by endpoint checks.

The 2026-07-21 v0.2 matrix remains preserved under
[`results/reference/`](../results/reference/) and its embedded manifest
protocols remain executable. New reports should use
[`results/v0.3/`](../results/v0.3/).
