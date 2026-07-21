# CLEAR-LeWM Evaluation Specification v0.1

This document is normative. Implementations may add diagnostics, but a result
may use a CLEAR-LeWM track name only if it follows the corresponding rules.

## 1. Shared task setup

All tracks use a future state from the same offline episode as the visual goal.
The default goal offset is 25 environment steps and the evaluation budget is 50
steps. A manifest fixes every `(episode_id, start_step, goal_step)` tuple before
any model is evaluated.

All methods, random baselines, and ablations in one comparison must use:

1. the same manifest;
2. the same environment and dataset fingerprints;
3. the same action bounds and planning budget;
4. explicitly recorded policy, solver, and environment seeds;
5. the same success criterion.

## 2. Dataset split

`official-compat` samples from all episodes, matching the upstream protocol.

`clear-id` also uses all episodes. It is intended for difficulty-controlled
evaluation of released checkpoints that were trained on the complete dataset,
and its results must be labeled in-distribution.

`clear-standard` and `clear-hard` assign exactly 20% of episode IDs to a
held-out split. Episode IDs are ranked by the first 64 bits of:

```text
SHA256("clear-lewm:{seed}:{episode_id}")
```

The lowest-ranked IDs form the held-out split. The split is deterministic and
independent of Python's process-randomized hash function. Training must exclude
these episodes for a held-out-generalization claim.

## 3. Pair construction and sampling

A start row is valid only when a row from the same episode exists exactly
`goal_offset` steps later.

`official-compat` samples uniformly without replacement over valid rows.
For exact upstream seed compatibility, it reproduces the upstream `N - 1`
choice range and therefore excludes the final valid global row. This behavior
is documented rather than silently repaired.

`clear-id`, `clear-standard`, and `clear-hard` first sample episode IDs uniformly
without replacement, then
sample one valid start row from each chosen episode. This prevents long episodes
from dominating evaluation. If more pairs than episodes are requested, episodes
are traversed in independently shuffled cycles.

These three tracks remove every pair satisfying the task's success condition at
the initial state. `clear-hard` additionally requires the following minimum
start-goal displacement:

| Task | Difficulty statistic | Minimum |
|---|---|---:|
| PushT | norm over the first four official state coordinates | 50.0 |
| Reacher | maximum absolute joint-angle difference | 0.25 rad |
| TwoRoom | Euclidean position distance | 32.0 |
| Cube | cube position distance | 0.08 m |

## 4. Success criteria

### PushT

Compatibility success follows the environment implementation: position-state
difference below 20 and block-angle difference below pi/9.

### Reacher

Compatibility success requires every target joint angle to be within 0.05 rad.

### TwoRoom

Compatibility success requires agent-goal Euclidean distance below 16 in the
environment's native coordinate system. Reports should also stratify same-room
and cross-room goals when wall metadata are available.

### OGBench-Cube

`official-compat` uses the upstream first-hit position test:

```text
norm(cube_position - target_position) <= 0.04 m
```

The CLEAR tracks require all of the following for five consecutive environment
steps:

```text
position error <= 0.04 m
quaternion geodesic error <= 15 degrees
```

Quaternion distance is `2 * acos(abs(dot(q, q_goal)))`; the absolute dot makes
the metric invariant to the equivalent representations `q` and `-q`.

The strict criterion evaluates stable target-pose attainment. It should not be
described as grasp success unless a separate grasp/contact requirement is added.

## 5. Metrics

The primary report contains:

- raw success rate and a 95% episode-bootstrap confidence interval;
- deterministic random-policy SR on the same manifest;
- excess over random in percentage points;
- paired bootstrap confidence interval for method minus random;
- normalized success `(method - random) / (100 - random)`;
- per-difficulty-bin SR;
- individual episode outcomes.

Training seeds and evaluation episodes are distinct uncertainty sources.
Aggregate across training seeds, but retain paired episode outcomes for each
seed.

## 6. Reproducibility record

A published result must retain:

- the manifest JSON;
- full dataset SHA256 or a publicly resolvable immutable dataset revision;
- LeWM, stable-worldmodel, and CLEAR-LeWM commits;
- Python and package lock files;
- checkpoint hash;
- solver parameters and all random seeds.

## 7. Compatibility statement

`official-compat` is intentionally not called a strict benchmark. It exists so
that prior LeWM-family results remain comparable. CLEAR tracks must be reported
separately and must never silently replace an official-compat number.
