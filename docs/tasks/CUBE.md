# Cube evaluation guide

> **Task contract:** move the physical cube to the target position and
> orientation, while treating the 24 proper rotations of an unmarked cube as
> physically equivalent. The robot arm and gripper are tools; their terminal
> configuration is not part of task success.

[Back to the four task contracts](../../README.md#what-v03-fixes) ·
[Normative v0.3 specification](../../EVALUATION_SPEC.md)

## What the policy sees and what the evaluator does

1. A manifest selects a start and a goal exactly 25 environment steps later
   from the same expert episode.
2. MuJoCo is reset to the recorded start state and the future RGB frame is
   supplied as the goal image. That image contains both the cube and the
   demonstrator's robot arm and gripper.
3. The evaluated policy receives pixels and emits its own actions; expert
   actions are not replayed.
4. MuJoCo advances the robot and cube under those actions.
5. The evaluator uses privileged cube pose only after each step to decide
   success. The policy never receives privileged pose, and the final robot or
   gripper pose is not scored.

The goal image is one visual representative of the desired cube pose. It is
not a requirement to reproduce the demonstrator's final robot configuration.

## Official and CLEAR manifest construction

| Manifest decision | Historical Official | CLEAR Moderate | CLEAR Strict |
|---|---|---|---|
| Pair source | same episode, `goal = start + 25` | same | same |
| Sampling unit | rows | episodes | episodes |
| Initially solved pair | retained | removed under position + symmetry-aware angle | removed under the tighter position + angle predicate |
| Goal orientation | present in data but ignored by the gate | included modulo cube symmetry | included modulo cube symmetry |
| Minimum cube translation | none | none beyond the unsolved gate | `>=0.08 m` |

> **MANIFEST GATE 1 — SAME-EPISODE FUTURE.** The target is a fixed future from
> the same physical demonstration, never a cross-episode image match.

> **MANIFEST GATE 2 — NOT PRE-SOLVED.** A start is removed only when it already
> satisfies both the mode's position tolerance and its symmetry-aware
> orientation tolerance.

> **MANIFEST GATE 3 — STRICT DISPLACEMENT.** Strict additionally requires at
> least `0.08 m` of cube translation. Moderate remains an attainable
> model-selection protocol and relies on the complete unsolved-pose gate.

The released seed-42 manifests contain 100 unique episodes per mode and zero
initial successes. Strict has a minimum observed cube translation of
`0.0823 m`.

## Official and CLEAR runtime success

| Runtime decision | Historical Official | CLEAR Moderate | CLEAR Strict |
|---|---|---|---|
| Cube position | `<=0.04 m` | `<=0.04 m` | `<=0.03 m` |
| Orientation | ignored | minimum geodesic error over 24 rotations `<=30 deg` | minimum geodesic error over 24 rotations `<=15 deg` |
| Robot/gripper terminal pose | not part of success | not part of success | not part of success |
| Temporal rule | first hit | 3 consecutive physical steps | 5 consecutive physical steps |

For current and target rotation matrices `R_c` and `R_g`, CLEAR evaluates

```text
min over S in the 24-element cube group: angle(R_c^T R_g S).
```

> **ROLLOUT GATE 4 — PHYSICAL ACTION EXECUTION.** The tested policy controls
> the MuJoCo robot. Success is not computed by comparing two offline rows.

> **SUCCESS GATE 5 — POSITION, SYMMETRY, AND HOLD.** A cube must be close in
> translation and in physical orientation for the complete hold. Raw
> quaternion equality is never required, and a different terminal arm or
> gripper pose cannot create either a success or a failure.

> **AUDIT GATE 6 — FIXED IDENTITY.** Model and random use the same manifest,
> dataset fingerprint, checkpoint record, solver budget, and environment
> fingerprint.

An optional settled-placement report may additionally require linear speed
below `0.05 m/s` and angular speed below `0.5 rad/s`. It is labeled separately
from primary SR.

## Benchmark success versus LeWM planning cost

> **INTERFACE BOUNDARY — BENCHMARK VS. PLANNER.** CLEAR evaluates the cube task
> after physically executing the policy. It does not change LeWM's internal
> latent objective or require another method to use a particular cost.

The standard full-image latent cost sees the cube together with the robot and
gripper. Consequently, its ranking may include residuals from a terminal robot
pose that is irrelevant to the task contract. CLEAR keeps this planning
surrogate unchanged for controlled comparison, but computes external success
only from the cube's symmetry-aware position and orientation. This separation
is deliberate: a benchmark defines what counts as completing the task, while
each algorithm remains responsible for choosing a useful optimization
surrogate.

The current audit does not attribute Cube performance to any single component
of the latent cost. It only guarantees that reported SR cannot improve or
degrade solely because the robot ends in a different pose from the expert.

## Why both released and naive predicates are incomplete

The released predicate ignores the target yaw and therefore grants success to
incorrect orientations. A naive raw-quaternion repair makes the opposite
mistake by separating orientations that are identical for an unmarked cube.
CLEAR evaluates the physical quotient: target pose modulo all 24 proper cube
rotations.

## Audited result

| Mode | LeWM | Random | Excess over random |
|---|---:|---:|---:|
| Moderate | **43%** | 4% | +39 pp |
| Strict | **18%** | 3% | +15 pp |

The symmetry-aware Moderate predicate recovers seven valid placements rejected
by raw quaternion matching on the same fixed pairs.

![Cube symmetry-aware trace](../../assets/task_gifs/cube.gif)

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.3/cube/strict-seed42-n100.json \
  --policy official/cube --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/cube_single_expert.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --output results/cube-strict.json
```
