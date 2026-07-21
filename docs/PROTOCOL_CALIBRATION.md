# Protocol calibration record

Calibration date: 2026-07-21. All values are SR on fixed 100-pair seed-42
manifests with official LeWM checkpoints and upstream `300 x 30` CEM. Thresholds
were accepted only when a converged model remained measurably above a seeded
random policy.

## Final four-task matrix

| Task | Official model/random | Moderate model/random | Strict model/random |
|---|---:|---:|---:|
| PushT | 89% / 7% | 74% / 0% | 42% / 0% |
| Reacher | 87% / 16% | 63% / 6% | 22% / 0% |
| TwoRoom | 85% / 30% | 70% / 17% | 41% / 1% |
| Cube | 62% / 47% | 36% / 3% | 17% / 2% |

## Reacher temporal calibration

Reacher is dynamic: the upstream controller often reaches the target joint
configuration with non-zero velocity. Requiring three or five consecutive
steps inside the original 0.05 rad band made both the official checkpoint and
random policy score 0%, so that criterion was rejected.

| Criterion | Official LeWM | Random | Decision |
|---|---:|---:|---|
| 0.05 rad, first hit | 39% | 5% | diagnostic only |
| 0.05 rad, hold 2 | 0% | 0% | reject: no model discrimination |
| 0.075 rad, hold 2 | 29% | 0% | Strict candidate |
| 0.10 rad, hold 2 | 56% | 5% | Moderate candidate |
| 0.035 rad, first hit | 14% | 1% | reject: loses temporal robustness |
| 0.025 rad, first hit | 6% | 1% | reject: weak discrimination |

Final manifests resampled after applying the selected initial-success filters
and difficulty rule. Their final SR is 63%/6% for Moderate and 22%/0% for
Strict.

## PushT strict calibration

All candidates below had 0% random SR on strict distance-filtered pairs.

| Position / angle / hold | Official LeWM | Decision |
|---|---:|---|
| 10 / 10 deg / 3 | 10% | reject: over-constrained |
| 10 / 10 deg / 2 | 30% | reject: weaker temporal rule |
| 15 / 15 deg / 3 | 49% | select |
| 20 / 20 deg / 5 | 39% | reject: geometry unchanged |
| 15 / 15 deg / 5 | 13% | reject: excessive hold |

After regenerating the final 15/15/hold-3 manifest, official LeWM scores 42%
and random scores 0%.

## Interpretation

The tiers are task-calibrated rather than forced to share one hold duration.
They remain monotonic within each task: Moderate is stricter than Official, and
Strict tightens Moderate geometry and/or pair difficulty without shortening its
task hold. Calibration used only official checkpoints and random policies; no
SICJEPA result influenced threshold selection.
