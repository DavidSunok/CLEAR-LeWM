# CLEAR-LeWM Evaluation Specification v0.5

This document is normative for the v0.5 Moderate protocol. A result may use
the `v0.5 Moderate` label only when it follows every rule below.

## 1. Scope

v0.5 defines a conservative benchmark correction:

- keep each task's upstream success semantics;
- remove start-goal pairs that are already successful at the start;
- sample episodes rather than over-weighting long trajectories;
- repair only demonstrated implementation defects;
- freeze model and random-policy evaluation to the same manifest.

The historical `official` protocol remains available for LeWM-compatible
reproduction. The v0.3 Strict protocol and its manifests remain immutable
historical artifacts; v0.5 does not silently reinterpret them.

## 2. Shared pair contract

Every Moderate pair satisfies all of the following:

1. `goal_row = start_row + 25` in the same offline episode;
2. the start fails the exact task predicate used during rollout;
3. sampling is episode-balanced, with one pair per episode when `n <=` the
   number of eligible episodes;
4. no additional minimum-displacement or difficulty threshold is applied;
5. seed 42 and `n=100` define the canonical release manifests.

The control budget is 50 environment steps. Expert actions between the start
and goal rows are never replayed during evaluation. The tested policy acts in
the simulator from the recorded start state.

## 3. Moderate task contracts

| Task | Pair gate | Runtime success | Temporal rule |
|---|---|---|---|
| PushT | remove starts already satisfying the official pose predicate | L2 over pusher and block positions `<20`, wrapped T angle `<20 deg` | first hit |
| Cube | remove starts with cube position error `<=0.04 m` | cube position error `<=0.04 m`; orientation and robot pose are not scored | first hit |
| Reacher | remove starts satisfying the corrected joint-topology error `<0.05 rad` | shoulder uses shortest periodic error; bounded wrist uses raw absolute error; max error `<0.05 rad` | first hit |
| TwoRoom | cross-room; start and goal disks clear; every transition in the 25-step source window legal | corrected swept-disk collision; endpoint distance `<16 px` | first hit |

### PushT

v0.5 intentionally retains the released five-dimensional goal-state meaning.
The position term contains both pusher and T-block coordinates, while the angle
term contains the wrapped T orientation. This matches the full goal-image
planning objective and avoids introducing a new block-only task.

### Cube

OGBench Cube defines success from object position only. Cube orientation,
gripper pose, and robot pose remain useful diagnostics but are not Moderate
success conditions. The 24-way cube-symmetry metric is retained in code for
archived v0.3 Strict manifests, not folded into v0.5 Moderate.

### Reacher

The DMC model has different joint topologies. The shoulder hinge is unbounded
and periodic, so its error is

```text
abs(atan2(sin(q - q_goal), cos(q - q_goal))).
```

The wrist hinge is range-limited, so wrapping it across `-pi/pi` would identify
physically distant states. Its error is ordinary absolute subtraction.

### TwoRoom

Cross-room is a sampling gate inherited from the canonical PLDM/DINO-WM task,
not an extra success condition. At runtime, the complete radius-7 agent disk is
resolved continuously against the wall and door. Success itself remains the
released endpoint distance `<16 px`.

The released TwoRoom dataset was generated with an endpoint collision defect.
Therefore v0.5 additionally requires `source_window_clean=true`: every one of
the 25 recorded transitions between start and goal must be legal under the
corrected swept-disk geometry. This prevents a corrupted demonstration window
from defining the benchmark goal. It does not replay those transitions during
evaluation.

## 4. Matched evaluation

All methods and baselines in one comparison must share:

1. manifest SHA-256 and dataset fingerprint;
2. task environment, action bounds, and 50-step budget;
3. policy, solver, and environment seeds;
4. planner samples, iterations, top-k, and solver batch size;
5. success protocol and evaluator commit.

Reference v0.5 runs use seed 42, `300 x 30` CEM, top-k 30, and solver batch
size 1. The paired random baseline uses the exact same 100 manifest pairs.

## 5. Reporting

The primary report includes success rate, episode-bootstrap 95% interval,
paired random SR, excess over random, all episode outcomes, checkpoint identity,
solver parameters, dataset fingerprint, environment fingerprint, and manifest
SHA-256. TwoRoom additionally records route and collision diagnostics even
though route validity is not a second success predicate in Moderate.

Training-seed uncertainty and evaluation-episode uncertainty are different.
Aggregate across training seeds when available while preserving paired episode
outcomes for each seed.

## 6. Compatibility

Every manifest embeds its complete `ProtocolSpec`. Older v0.1-v0.3 manifests
therefore continue to execute with their original thresholds, hold durations,
and task semantics after v0.5 is installed. Published artifacts must never be
edited in place.
