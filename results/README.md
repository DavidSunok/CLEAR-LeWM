# Reference results

[`v0.5/`](v0.5/) is the current reference set. It contains only paired random
baselines and the four pinned official high-epoch LeWM checkpoints evaluated
under v0.5 Moderate and Strict.

| Mode | Coverage | Purpose |
|---|---|---|
| Moderate | official and random seeds 0/1/42 | minimally repaired LeWM-compatible comparison |
| Strict | official and random seeds 0/1/42 | task-semantic precision comparison |

Every model result records raw SR, confidence intervals, excess over its paired
random trace, all episode outcomes, manifest and checkpoint hashes, solver
budget, environment fingerprints, and strict 303/303 tensor loading.

[`reference/`](reference/) retains earlier compatibility artifacts. Previous
releases remain reproducible from their Git tags; their values are not renamed
or mixed into v0.5.
