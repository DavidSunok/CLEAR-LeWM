# CLEAR-LeWM Evaluation Specification v0.2

This document is normative. A result may use a CLEAR-LeWM tier name only when
it follows the corresponding rules below.

## 1. Shared task setup

Every tier uses a future state from the same offline episode as the visual goal.
The default goal offset is 25 environment steps and the control budget is 50
steps. A versioned manifest fixes every
`(episode_id, start_step, goal_step)` tuple before any policy is evaluated.

All methods and baselines in one comparison must use the same:

1. manifest and dataset fingerprint;
2. environment, action bounds, and evaluation budget;
3. policy, solver, and environment seeds;
4. planning horizon, samples, iterations, and top-k;
5. task success tier.

## 2. Three success tiers

### Official

`official` reproduces upstream LeWM evaluation for historical comparison:

- row-uniform sampling over all valid start rows;
- initially successful pairs retained;
- the upstream `N - 1` sampling range retained;
- original task thresholds and first-hit termination;
- deterministic policy seeding added and recorded.

This tier is compatibility evidence, not a strict benchmark.

### Moderate

`moderate` samples episodes uniformly, removes pairs already successful at the
start, and requires a task-calibrated stable target predicate. Hold duration is
two steps for dynamic Reacher and three for the other tasks.

### Strict

`strict` uses episode-balanced sampling, removes initially successful pairs,
requires a larger start-goal displacement, and applies tighter task tolerances
and task-specific hold durations.

| Task | Official | Moderate | Strict |
|---|---|---|---|
| PushT | position `<20`, angle `<20 deg`, first hit | same geometry, hold 3 | position `<15`, angle `<15 deg`, hold 3 |
| Reacher | every joint `<0.05 rad`, first hit | every joint `<0.10 rad`, hold 2 | every joint `<0.075 rad`, hold 2 |
| TwoRoom | distance `<16`, first hit | distance `<12`, hold 3 | distance `<8`, hold 5 |
| Cube | position `<=0.04 m`, first hit | position `<=0.04 m`, orientation `<=30 deg`, hold 3 | position `<=0.03 m`, orientation `<=15 deg`, hold 5 |

Cube orientation uses quaternion geodesic distance
`2 * acos(abs(dot(q, q_goal)))`, which is invariant to the equivalent
representations `q` and `-q`. The criterion tests stable target-pose attainment;
it is not called grasp success because contact is not required.

## 3. Pair construction

A start row is valid only when a row from the same episode exists exactly 25
steps later. `official` samples valid rows without replacement. `moderate` and
`strict` sample episode IDs uniformly, then one valid start per selected
episode. When more pairs than episodes are requested, independently shuffled
episode cycles are used.

Strict minimum start-goal displacement is:

| Task | Statistic | Minimum |
|---|---|---:|
| PushT | norm over the first four official state coordinates | 50.0 |
| Reacher | maximum absolute joint-angle difference | 0.25 rad |
| TwoRoom | Euclidean position distance | 32.0 |
| Cube | cube position distance | 0.08 m |

## 4. Data split is orthogonal

Success rigor and trajectory generalization answer different questions.

- `--split all` is the default in-distribution setting and is valid for released
  official checkpoints trained on the full public dataset.
- `--split heldout` deterministically assigns 20% of episode IDs to evaluation.
  It is a held-out claim only if those IDs were excluded from training.

Episode IDs are ranked by the first 64 bits of
`SHA256("clear-lewm:{seed}:{episode_id}")`; the lowest-ranked IDs are held out.
An official full-data checkpoint may be measured on those rows, but that number
must not be labeled held-out generalization.

The physical training reader is governed by [`DATA_SPEC.md`](DATA_SPEC.md).
Evaluation manifests always reference the canonical HDF5 row space.

## 5. Metrics

The primary report contains:

- raw success rate and 95% episode-bootstrap confidence interval;
- deterministic random-policy SR on the identical manifest;
- excess over random in percentage points;
- paired bootstrap interval for method minus random;
- normalized success `(method - random) / (100 - random)`;
- per-difficulty-bin SR and individual episode outcomes.

Random normalization does not replace a meaningful success predicate. Report
all three tiers separately; never mix their numbers in one ranking column.

## 6. Reproducibility record

A published result retains the manifest JSON, dataset revision or SHA-256,
training-data record, LeWM/stable-worldmodel/CLEAR-LeWM commits, checkpoint
SHA-256, solver parameters, package lock, and every random seed.

Training-seed and evaluation-episode uncertainty are different. Aggregate
across training seeds, while retaining paired per-episode outcomes per seed.

## 7. v0.1 compatibility

`official-compat`, `clear-id`, `clear-standard`, and `clear-hard` remain accepted
so v0.1 manifests are executable. New reports should use `official`, `moderate`,
and `strict`; old names are not silently reinterpreted.
