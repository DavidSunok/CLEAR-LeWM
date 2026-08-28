# CLEAR-LeWM v0.8: RTX 4090 full matrix audit

Status: all 84 result JSON files passed identity and protocol checks.

- GPU: NVIDIA GeForce RTX 4090
- seeds: 0, 1, 42
- episodes: 100 per seed
- inference: pure CEM, actor warm-start off
- solver: 300 samples x 30 iterations, top-k 30, batch size 1
- checkpoint loading: strict, no missing or unexpected keys
- paired random: same task, mode, seed, and environment fingerprints

## Official LeWM

| Task | Mode | Seed 0 / 1 / 42 | v0.8 4090 mean +/- s.d. | Random | v0.5 H200 | Delta | Mean trace flips |
|---|---|---:|---:|---:|---:|---:|---:|
| pusht | moderate | 87 / 87 / 86 | 86.67 +/- 0.58% | 4.00% | 86.33% | +0.33 pp | 5.00/100 |
| pusht | strict | 65 / 76 / 71 | 70.67 +/- 5.51% | 5.00% | 70.33% | +0.33 pp | 5.00/100 |
| cube | moderate | 46 / 52 / 53 | 50.33 +/- 3.79% | 15.67% | 50.33% | +0.00 pp | 9.33/100 |
| cube | strict | 20 / 27 / 18 | 21.67 +/- 4.73% | 6.00% | 26.33% | -4.67 pp | 10.00/100 |
| reacher | moderate | 79 / 80 / 80 | 79.67 +/- 0.58% | 7.33% | 46.00% | +33.67 pp | 43.67/100 |
| reacher | strict | 86 / 87 / 88 | 87.00 +/- 1.00% | 8.00% | 43.00% | +44.00 pp | 48.67/100 |
| tworoom | moderate | 89 / 81 / 79 | 83.00 +/- 5.29% | 6.67% | 84.00% | -1.00 pp | 7.00/100 |
| tworoom | strict | 50 / 55 / 49 | 51.33 +/- 3.21% | 1.67% | 58.33% | -7.00 pp | 23.67/100 |

Reacher v0.8 disables the underlying dm-control task termination before CLEAR scoring, so its v0.8-v0.5 delta is a semantic correction, not a hardware-variance estimate. The other three tasks retain the same scoring semantics and are the appropriate hardware/runtime comparison.

## Related public checkpoints

These models have no released Reacher checkpoint.

| Model | Task | Mode | Seed 0 / 1 / 42 | 4090 mean +/- s.d. | Random | Excess |
|---|---|---|---:|---:|---:|---:|
| DINOv2 No-Proprio LeWM | pusht | moderate | 10 / 3 / 8 | 7.00 +/- 3.61% | 4.00% | +3.00 pp |
| DINOv2 No-Proprio LeWM | pusht | strict | 10 / 9 / 7 | 8.67 +/- 1.53% | 5.00% | +3.67 pp |
| DINOv2 No-Proprio LeWM | cube | moderate | 47 / 42 / 45 | 44.67 +/- 2.52% | 15.67% | +29.00 pp |
| DINOv2 No-Proprio LeWM | cube | strict | 13 / 10 / 16 | 13.00 +/- 3.00% | 6.00% | +7.00 pp |
| DINOv2 No-Proprio LeWM | tworoom | moderate | 39 / 38 / 54 | 43.67 +/- 8.96% | 6.67% | +37.00 pp |
| DINOv2 No-Proprio LeWM | tworoom | strict | 28 / 24 / 25 | 25.67 +/- 2.08% | 1.67% | +24.00 pp |
| GCBC Joint LeWM | pusht | moderate | 5 / 2 / 9 | 5.33 +/- 3.51% | 4.00% | +1.33 pp |
| GCBC Joint LeWM | pusht | strict | 7 / 6 / 9 | 7.33 +/- 1.53% | 5.00% | +2.33 pp |
| GCBC Joint LeWM | cube | moderate | 15 / 25 / 13 | 17.67 +/- 6.43% | 15.67% | +2.00 pp |
| GCBC Joint LeWM | cube | strict | 3 / 6 / 2 | 3.67 +/- 2.08% | 6.00% | -2.33 pp |
| GCBC Joint LeWM | tworoom | moderate | 19 / 16 / 13 | 16.00 +/- 3.00% | 6.67% | +9.33 pp |
| GCBC Joint LeWM | tworoom | strict | 6 / 8 / 6 | 6.67 +/- 1.15% | 1.67% | +5.00 pp |

## Issue #3 comparison

| Task | This audit, strict 4090 | Issue #3, strict 4090D | Delta | Interpretation |
|---|---:|---:|---:|---|
| pusht | 70.67 +/- 5.51% | 70.67 +/- 5.13% | +0.00 pp | same task semantics |
| cube | 21.67 +/- 4.73% | 21.67 +/- 4.73% | +0.00 pp | same task semantics |
| reacher | 87.00 +/- 1.00% | 40.33 +/- 7.51% | +46.67 pp | not a hardware comparison; v0.8 fixes Reacher termination |
| tworoom | 51.33 +/- 3.21% | 51.67 +/- 2.89% | -0.33 pp | same task semantics |

## Versioning decision

`results/v0.5/` remains immutable historical evidence. Corrected RTX 4090 results belong under `results/v0.8/`; README tables and media should point to v0.8 while retaining an explicit v0.5 archive link.
