# CLEAR-LeWM Evaluation Specification v0.5

This document is normative for both CLEAR-LeWM v0.5 protocols. A result may
use the `v0.5 Moderate` or `v0.5 Strict` label only when it uses the matching
checked-in manifest and every rule below.

## Protocol intent

**Moderate is a minimal compatibility correction.** It removes initially
solved evaluation pairs, balances episodes, repairs the Reacher angle-topology
bug, and repairs the TwoRoom collision rewrite and contaminated source
windows. PushT and Cube otherwise retain the released LeWM task predicates.

**Strict is a task-semantic precision audit.** It inherits Moderate data
hygiene and corrected physics, then evaluates only the task-relevant object or
endpoint with tighter geometry and short persistence requirements. It is the
preferred mode for claims about fine manipulation or precise navigation.

The modes answer different questions. Moderate asks whether a method improves
LeWM under a minimally corrected version of its intended benchmark. Strict
asks whether the resulting behavior precisely completes the task itself.

## Shared pair contract

Both modes use:

1. `goal = start + 25` from the same recorded episode;
2. episode-balanced sampling from the full released evaluation snapshot;
3. one canonical candidate per selected episode;
4. removal of pairs already solved under that mode's rollout predicate;
5. 100 fixed pairs and a recorded policy seed per canonical manifest;
6. a 50-step closed-loop control budget.

No mode replays expert actions. Privileged state is read only by the external
evaluator after the evaluated policy advances the simulator.

## Moderate: minimal evaluator repair

### PushT

- Preserve the released five-dimensional goal-state meaning.
- Success requires `||state[:4] - goal[:4]||_2 < 20 px` and shortest wrapped
  T-block angle error `< 20 deg`.
- The first valid hit succeeds; no hold or minimum displacement is added.

Moderate therefore scores both the pusher and T block, matching the goal image
and upstream latent-MSE planning target as closely as possible.

### Cube

- Follow the released OGBench success predicate.
- Success requires cube-center distance `<= 0.04 m`.
- Cube orientation and terminal robot/gripper pose are not scored.
- The first valid hit succeeds.

### Reacher

- Preserve the released joint-configuration target and `0.05 rad` threshold.
- The unbounded shoulder uses shortest periodic error.
- The physically bounded wrist uses raw absolute error.
- Success requires the maximum corrected joint error `< 0.05 rad`.
- The first valid hit succeeds.

### TwoRoom

- Start and goal must lie in opposite rooms and both full radius-7 disks must
  be collision-free.
- Every transition in the recorded 25-step source window must be legal under
  continuous swept-disk geometry.
- Runtime motion is resolved as a complete swept disk with full-radius door
  clearance; the endpoint-only collision rewrite is not used.
- Success requires endpoint distance `< 16 px`; the first valid hit succeeds.
- Route validity and legal crossings are recorded for audit, but corrected
  physics already prevents wall traversal and they are not extra Moderate
  success predicates.

## Strict: task-semantic precision

Strict uses the same data hygiene and corrected simulator execution as
Moderate, with the following task-semantic gates.

### PushT

- Score only the T block; final pusher location is irrelevant.
- T-block position error must be `< 10 px`.
- Shortest wrapped T-block angle error must be `< 10 deg`.
- Both conditions must hold for 3 consecutive simulator steps.

### Cube

- Score only the cube; terminal robot/gripper pose is irrelevant.
- Cube-center distance must be `<= 0.03 m`.
- Orientation error must be `<= 15 deg`, minimized over all 24 proper cube
  rotations.
- Both conditions must hold for 3 consecutive simulator steps.

### Reacher

- Score the physical fingertip endpoint rather than a redundant joint pose.
- Fingertip distance to the endpoint induced by the goal configuration must be
  `<= 0.01 m`.
- The endpoint condition must hold for 2 consecutive simulator steps.

### TwoRoom

- Use only clean cross-room source windows and corrected swept-disk physics.
- Endpoint distance must be `< 8 px`.
- The complete route must remain valid.
- At least one legal room crossing must occur and the agent must reach the
  goal side of the wall.
- Success is first hit after all conditions hold; no redundant hold is added.

## Reference inference contract

Official LeWM reference runs use the pinned high-epoch checkpoints, 300 CEM
samples, 30 CEM iterations, top-k 30, and solver batch size 1. Model and random
must use the identical manifest and policy seed. Batch 16 is a development
throughput mode and is not numerically equivalent.

Representation-only planning comparisons must pass `--actor-warmstart off`.
Action-head or custom-runtime evaluations must state their inference mode and
retain the runtime source hashes written by CLEAR-LeWM.

## Version and provenance

Canonical manifests live at:

```text
manifests/v0.5/<task>/<moderate|strict>-seed<seed>-n100.json
```

Each manifest embeds its complete `ProtocolSpec`, dataset fingerprint, selected
pair IDs, and policy seed. Each result records the manifest SHA-256, all episode
outcomes, criterion, solver, checkpoint audit, environment fingerprints, and
task-specific diagnostics. A change to sampling, success semantics, physics,
or aggregation requires a new benchmark version.

Previous public protocols remain reproducible from their Git release tags;
their numbers are not relabeled as v0.5 results.
