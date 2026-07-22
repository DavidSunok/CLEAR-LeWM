# Reference results

`v0.3/` contains the current task-semantic deterministic random and official
LeWM outputs. `reference/` preserves the v0.2 three-tier matrix.

| Task | Moderate LeWM / Random | Strict LeWM / Random |
|---|---:|---:|
| PushT | 93% / 7% | 79% / 2% |
| Cube | 43% / 4% | 18% / 3% |
| Reacher | 90% / 17% | 36% / 4% |
| TwoRoom | 61% / 2% | 24% / 0% |

Every v0.3 result uses 100 episodes, manifest and policy seed 42, upstream CEM
with 300 samples and 30 iterations, and solver batch size 1.

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

The v0.1 `clear-id` random outputs remain for backward reproducibility. Binary
model weights and datasets remain outside Git; revision and SHA-256 identities
are stored in [`../checkpoints/`](../checkpoints/).
