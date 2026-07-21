# Audit findings for the public LeWM evaluation protocol

Audit date: 2026-07-21.

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

## CLEAR-ID reference floor

The checked-in `clear-id` manifests use 100 distinct episodes, remove initially
solved pairs, and seed both manifest selection and random actions with 42. Cube
additionally requires position within 0.04 m, quaternion error within 15 degrees,
and five consecutive successful steps.

| Task | CLEAR-ID Random SR | 95% episode-bootstrap CI |
|---|---:|---:|
| PushT | 3% | [0%, 7%] |
| Reacher | 9% | [4%, 15%] |
| TwoRoom | 22% | [14%, 30%] |
| OGBench-Cube | 1% | [0%, 3%] |

The Cube floor falls from approximately 49% under the original setup to 1%
under the strict in-distribution track. TwoRoom retains a material random-walk
floor even after initially solved pairs are removed.

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
