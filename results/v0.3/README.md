# CLEAR-LeWM v0.3 audited results

All files use 100 fixed pairs, manifest and policy seed 42, `300 x 30` CEM,
top-k 30, and solver batch size 1.

| Task | Moderate model/random | Strict model/random |
|---|---:|---:|
| PushT | 93% / 7% | 79% / 2% |
| Cube | 43% / 4% | 18% / 3% |
| Reacher | 90% / 17% | 36% / 4% |
| TwoRoom | 61% / 2% | 24% / 0% |

Each model output contains checkpoint source and runtime hashes, strict tensor
load audit, environment fingerprint, manifest SHA-256, paired random metrics,
and all episode outcomes. TwoRoom outputs additionally contain per-episode
route topology and continuous-collision diagnostics.

