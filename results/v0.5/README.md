# CLEAR-LeWM v0.5 Moderate results

The files in this directory pair the official high-epoch LeWM checkpoint and a
deterministic random policy on each canonical v0.5 Moderate manifest. All runs
use seed 42, 100 episodes, a 50-step control budget, and solver batch size 1.
Official LeWM runs use `300 x 30` CEM with top-k 30.

| Task | Official LeWM | Random | Excess over random |
|---|---:|---:|---:|
| PushT | 88% | 3% | +85 pp |
| Cube | 51% | 15% | +36 pp |
| Reacher | 40% | 5% | +35 pp |
| TwoRoom | 81% | 6% | +75 pp |

Every model result contains strict checkpoint tensor audit, pinned upstream
source revision and hashes, environment fingerprints, manifest SHA-256,
solver settings, paired-random metrics, and all 100 episode outcomes.
