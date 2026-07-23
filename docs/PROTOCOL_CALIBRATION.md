# CLEAR-LeWM v0.5 protocol calibration

Calibration dates: 2026-07-22 to 2026-07-23. All model values use the pinned
official high-epoch LeWM checkpoints, `300 x 30` CEM, top-k 30, solver batch
size 1, and 100 fixed pairs per manifest.

## Moderate: minimal compatibility correction

Moderate uses seeds 0, 1, and 42. Values are mean +/- sample standard
deviation.

| Task | Official LeWM | Paired random |
|---|---:|---:|
| PushT | 86.33 +/- 2.08% | 4.00 +/- 1.00% |
| Cube | 50.33 +/- 4.04% | 15.67 +/- 6.03% |
| Reacher | 46.00 +/- 5.57% | 4.33 +/- 1.15% |
| TwoRoom | 84.00 +/- 3.00% | 6.67 +/- 1.15% |

Moderate was accepted only after confirming that it changes no more than the
released benchmark requires:

- all tasks remove initially solved pairs and sample episodes uniformly;
- PushT keeps the complete released pusher-plus-block goal state;
- Cube keeps OGBench's 4 cm object-position success;
- Reacher wraps only the unbounded shoulder and leaves the bounded wrist raw;
- TwoRoom restores cross-room sampling, rejects polluted source windows, and
  replaces endpoint-only collision with continuous swept-disk physics while
  retaining the released 16 px endpoint predicate.

## Strict: task-semantic precision

Strict uses seeds 0, 1, and 42. Values are mean +/- sample standard deviation.

| Task | Official LeWM | Paired random |
|---|---:|---:|
| PushT | 70.33 +/- 4.04% | 5.00 +/- 1.73% |
| Cube | 26.33 +/- 1.53% | 6.00 +/- 2.65% |
| Reacher | 43.00 +/- 7.21% | 5.00 +/- 3.46% |
| TwoRoom | 58.33 +/- 2.31% | 1.67 +/- 2.89% |

Strict thresholds were selected to express each task's physical endpoint
without collapsing the converged official checkpoint to zero:

- PushT: T-block `10 px / 10 deg`, hold 3;
- Cube: cube center `3 cm` plus 24-fold orientation `15 deg`, hold 3;
- Reacher: physical fingertip endpoint `1 cm`, hold 2;
- TwoRoom: legal cross-room route, goal side reached, endpoint `8 px`.

## Selection guardrails

Every accepted threshold satisfies:

1. selected pairs are initially unsolved under the same rollout predicate;
2. model and random use identical manifests and policy seeds;
3. the official checkpoint remains measurably above random;
4. task-equivalent states are not separated by representation artifacts;
5. TwoRoom wall geometry cannot be bypassed by endpoint checks;
6. task-irrelevant robot pose is excluded only in Strict, where the semantic
   change is explicit.

Previous protocols are preserved by their Git release tags; their numbers are
not renamed or pooled with v0.5.
