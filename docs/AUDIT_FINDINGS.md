# Audit findings for LeWM evaluation

Audit dates: 2026-07-21 to 2026-07-23. The analysis used the four public LeWM
dataset snapshots and `stable-worldmodel==0.1.0`.

## Why a corrected benchmark is needed

Under the historical sampling/evaluator stack, many `start, start+25` pairs
are already inside the task threshold and random policies can retain high SR.

| Task | Initially within historical threshold | Random SR, three runs | Mean |
|---|---:|---:|---:|
| PushT | 0.22% | 6%, 5%, 1% | 4.0% |
| Reacher | 0.57% | 14%, 15%, 10% | 13.0% |
| TwoRoom | 8.82% | 25%, 25%, 28% | 26.0% |
| OGBench-Cube | 38.38% | 48%, 43%, 56% | 49.0% |

Cube's high floor is consistent with long pre-grasp periods in which the image
changes while the cube remains within 4 cm. TwoRoom additionally suffers from
an environment rewrite that can admit a requested segment through solid wall
geometry even when its endpoint is legal.

## v0.5 findings

Moderate removes pre-solved pairs and repairs only demonstrated evaluator
defects. Strict then asks whether the task-relevant object or endpoint is
precisely complete.

| Task | Moderate LeWM/random, 3-seed mean | Strict LeWM/random, 3-seed mean |
|---|---:|---:|
| PushT | 86.33% / 4.00% | 70.33% / 5.00% |
| Cube | 50.33% / 15.67% | 26.33% / 6.00% |
| Reacher | 46.00% / 4.33% | 43.00% / 5.00% |
| TwoRoom | 84.00% / 6.67% | 58.33% / 1.67% |

The strict TwoRoom evaluator recorded zero invalid routes in all six checked-in
official/random runs for seeds 0, 1, and 42. It also requires a legal room
crossing and arrival on the goal side, preventing endpoint proximity from
crediting a topologically invalid trajectory.

## Cube interpretation

The upstream task checks cube position within 4 cm and does not score target
orientation or terminal gripper pose. Across 1,760,000 valid Cube pairs,
38.38% have displacement at most 4 cm. Moderate preserves that official task
definition after removing pre-solved starts. Strict separately claims precise
pose completion using the 24 proper cube rotations.

## Upstream reports relevant to interpretation

- [Cube task: success evaluation, issue #89](https://github.com/lucas-maes/le-wm/issues/89)
  reports rollouts counted as successful despite incorrect or incomplete grasping.
- [Evaluation dataset clarification, issue #90](https://github.com/lucas-maes/le-wm/issues/90)
  asks whether evaluation uses held-out episodes or the training dataset.
- [Evaluation sampling off-by-one, issue #77](https://github.com/lucas-maes/le-wm/issues/77)
  identifies exclusion of the final valid global row.

## Interpretation

The original protocol remains useful for reproducing historical papers when
every method uses identical code and pairs. For new comparisons, Moderate is
the least disruptive corrected baseline; Strict is the more discriminative
task-semantic metric. Reports should name the mode rather than averaging the
two into an unlabeled score.
