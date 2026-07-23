# CLEAR-LeWM v0.5 official reference results

This directory contains only deterministic random baselines and reevaluations
of the four pinned official high-epoch LeWM checkpoints. No private or
project-specific checkpoint result is included.

All official runs use 100 episodes, a 50-step control budget, `300 x 30` CEM,
top-k 30, solver batch size 1, and strict 303/303 tensor loading. Model and
random share the exact manifest and policy seed.

## Moderate: minimal repair

Official checkpoint results are reported on the canonical seed-42 manifest.

| Task | Official LeWM | Paired random | Excess |
|---|---:|---:|---:|
| PushT | 88% | 3% | +85 pp |
| Cube | 51% | 15% | +36 pp |
| Reacher | 40% | 5% | +35 pp |
| TwoRoom | 81% | 6% | +75 pp |

## Strict: task-semantic precision

Strict results use three complete manifest/policy seeds. Values below are
mean +/- sample standard deviation across seeds 0, 1, and 42.

| Task | Official LeWM | Paired random | Mean excess |
|---|---:|---:|---:|
| PushT | 70.33 +/- 4.04% | 5.00 +/- 1.73% | +65.33 pp |
| Cube | 26.33 +/- 1.53% | 6.00 +/- 2.65% | +20.33 pp |
| Reacher | 43.00 +/- 7.21% | 5.00 +/- 3.46% | +38.00 pp |
| TwoRoom | 58.33 +/- 2.31% | 1.67 +/- 2.89% | +56.67 pp |

Every result stores all episode outcomes, manifest SHA-256, task criterion,
solver settings, paired-random metrics, environment fingerprints, checkpoint
provenance, and strict state-dict audit. TwoRoom additionally stores complete
route, wall-contact, legal-crossing, and goal-side diagnostics.
