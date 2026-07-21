# Reference results

`reference/` contains the v0.2 deterministic random and official LeWM outputs
for four tasks and three tiers. Every cell uses 100 episodes, manifest seed 42,
policy seed 42, and upstream CEM with 300 samples and 30 iterations.

| Task | Official LeWM / Random | Moderate LeWM / Random | Strict LeWM / Random |
|---|---:|---:|---:|
| PushT | 89% / 7% | 74% / 0% | 42% / 0% |
| Reacher | 87% / 16% | 63% / 6% | 22% / 0% |
| TwoRoom | 85% / 30% | 70% / 17% | 41% / 1% |
| OGBench-Cube | 62% / 47% | 36% / 3% | 17% / 2% |

Each model result records raw SR and bootstrap CI, excess over random, paired
gain CI, normalized success, per-episode outcomes, manifest SHA-256, task
criterion, solver budget, runtime checkpoint SHA-256, official source revision,
source weight SHA-256, and the certified 303-tensor load count.

The v0.1 `clear-id` random outputs remain for backward reproducibility. They are
not used in the v0.2 table. Large videos, model checkpoints, datasets, and local
experiment logs are intentionally excluded from Git.
