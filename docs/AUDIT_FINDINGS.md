# Audit findings for the public LeWM evaluation protocol

Audit dates: 2026-07-21 initial protocol audit; 2026-07-22 task-semantic and
TwoRoom topology audit.

The analysis used the four public LeWM dataset snapshots available locally and
`stable-worldmodel==0.1.0`. Initial-success rates were computed over every valid
`(t, t+25)` pair. Random-policy values used 100 episodes for each of three fixed
manifest seeds and separately seeded action spaces.

| Task | Initially within success threshold | Random SR, three runs | Mean |
|---|---:|---:|---:|
| PushT | 0.22% | 6%, 5%, 1% | 4.0% |
| Reacher | 0.57% | 14%, 15%, 10% | 13.0% |
| TwoRoom | 8.82% | 25%, 25%, 28% | 26.0% |
| OGBench-Cube | 38.38% | 48%, 43%, 56% | 49.0% |

The LeWM paper reports Random SR of 2% on PushT, 10% on Reacher, 0% on
TwoRoom, and 48% on Cube. The current Cube result reproduces the reported high
floor. The current TwoRoom stack does not reproduce the paper's zero random
baseline, indicating dependency, data, environment, or protocol drift that
should be resolved with immutable version records.

## v0.3 task-semantic reference

v0.3 corrects four independent mechanisms rather than applying one generic
threshold change: PushT object semantics, Cube rotational symmetry, Reacher
periodic first-hit semantics, and TwoRoom continuous route validity.

| Task | Moderate LeWM/random | Strict LeWM/random |
|---|---:|---:|
| PushT | 93% / 7% | 79% / 2% |
| Cube | 43% / 4% | 18% / 3% |
| Reacher | 90% / 17% | 36% / 4% |
| TwoRoom | 61% / 2% | 24% / 0% |

All 200 PushT observation rollouts remained inside the simulated workspace.
Cube and Reacher produced no non-finite states. The critical physical defect was
TwoRoom: 87/100 original cross-room Strict trajectories were route-invalid.
Swept-disk evaluation reduced that count to zero while preserving 24% legal SR.

## v0.2 calibrated reference

The checked-in v0.2 manifests use 100 fixed pairs and deterministic policy seed
42. Moderate and Strict sample episodes uniformly and remove pairs already
successful under their task criterion.

| Task | Official LeWM/random | Moderate LeWM/random | Strict LeWM/random |
|---|---:|---:|---:|
| PushT | 89% / 7% | 74% / 0% | 42% / 0% |
| Reacher | 87% / 16% | 63% / 6% | 22% / 0% |
| TwoRoom | 85% / 30% | 70% / 17% | 41% / 1% |
| OGBench-Cube | 62% / 47% | 36% / 3% | 17% / 2% |

The random floor falls sharply without making converged official checkpoints
score zero. TwoRoom Moderate deliberately remains a mild correction and still
has a 17% random floor; reports should use its paired 53pp model gain or select
Strict when a near-zero floor is required. Reacher and PushT hold durations were
calibrated against both official checkpoints and random policies; see
[`PROTOCOL_CALIBRATION.md`](PROTOCOL_CALIBRATION.md).

## Cube mechanism

The upstream Cube configuration passes both target position and target
quaternion to the environment. The success implementation checks only cube
position within 0.04 m. It does not use target orientation, gripper contact, or
trajectory-level grasp information.

Across all 1,760,000 valid Cube start-goal pairs:

- 38.38% have cube displacement at most 0.04 m;
- 33.62% have displacement at most 0.02 m;
- 30.78% have displacement at most 0.01 m.

These pairs commonly correspond to pre-grasp arm motion during which the goal
image changes but the cube remains stationary. The visual planning objective and
the environment success predicate therefore measure different aspects of the
goal.

## Upstream reports relevant to interpretation

- [Cube task: success evaluation, issue #89](https://github.com/lucas-maes/le-wm/issues/89)
  reports rollouts counted as successful despite incorrect or incomplete grasping.
- [Evaluation dataset clarification, issue #90](https://github.com/lucas-maes/le-wm/issues/90)
  asks whether evaluation uses held-out episodes or the training dataset.
- [Evaluation sampling off-by-one, issue #77](https://github.com/lucas-maes/le-wm/issues/77)
  identifies exclusion of the final valid global row. This bug is real but has
  negligible numerical impact compared with the protocol-level effects above.

## Interpretation

The official protocol remains useful for historical method-to-method comparison
when every method uses identical pairs and code. It should be described as
in-distribution future-state planning. Cube raw SR alone is not evidence of a
correct grasp-and-place sequence, and a four-task raw mean gives Cube's high
floor the same weight as lower-floor tasks.
