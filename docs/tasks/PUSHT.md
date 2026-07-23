# PushT evaluation guide

> **Task contract:** move the T-shaped block to the target planar pose. The
> circular pusher is the tool used to solve the task, not part of the completed
> object state.

[Back to the four task contracts](../../README.md#what-v03-fixes) ·
[Normative v0.3 specification](../../EVALUATION_SPEC.md)

## What the policy sees and what the evaluator does

1. A manifest selects a start and a goal exactly 25 environment steps later
   from the same offline episode.
2. The simulator is reset to the recorded start state. The full future RGB
   frame is supplied as the goal image, so it still contains both the T block
   and the pusher.
3. The evaluated policy receives pixels and emits its own actions. Expert
   actions are not replayed.
4. Each action is applied by the PushT PD controller and advanced through the
   Pymunk contact simulation.
5. The evaluator reads privileged state only after each physical step to judge
   the T-block pose. Privileged state is never an input to the policy.

The goal image is one visual representative of a successful object pose. CLEAR
does not require the policy to reproduce the demonstrator's final pusher
location.

## Official and CLEAR manifest construction

| Manifest decision | Historical Official | CLEAR Moderate | CLEAR Strict |
|---|---|---|---|
| Pair source | same episode, `goal = start + 25` | same | same |
| Sampling unit | rows | episodes | episodes |
| Initially solved pair | retained | removed using the Moderate block-pose predicate | removed using the Strict block-pose predicate |
| Difficulty variable | none; pusher motion can make the pair appear nontrivial | T-block translation | T-block translation |
| Minimum task motion | none | `>=25 px` block translation | `>=50 px` block translation |
| Released off-by-one range | preserved | corrected | corrected |

> **MANIFEST GATE 1 — SAME-EPISODE FUTURE.** The goal must exist at the exact
> `+25` offset in the same episode.

> **MANIFEST GATE 2 — NOT PRE-SOLVED.** The start T-block pose must fail the
> same position-and-angle predicate later used during rollout.

> **MANIFEST GATE 3 — REAL OBJECT MOTION.** Pusher-only motion cannot create a
> Moderate or Strict pair: the T block itself must translate by at least 25 or
> 50 pixels.

The released seed-42 manifests contain 100 unique episodes per mode, zero
initially successful pairs, and minimum observed block translations of
`25.182 px` (Moderate) and `50.143 px` (Strict).

## Official and CLEAR runtime success

The state layout begins with
`[pusher_x, pusher_y, block_x, block_y, block_angle, ...]`.

| Runtime decision | Historical Official | CLEAR Moderate | CLEAR Strict |
|---|---|---|---|
| Position error | L2 over `state[:4]`: pusher **and** block | L2 over block `state[2:4]` | L2 over block `state[2:4]` |
| Position threshold | `<20 px` | `<20 px` | `<15 px` |
| Block-angle threshold | `<20 deg` | `<20 deg` | `<15 deg` |
| Temporal rule | first hit | 3 consecutive physical steps | 5 consecutive physical steps |

> **ROLLOUT GATE 4 — PHYSICAL ACTION EXECUTION.** A score is produced only
> after the tested policy's actions have been executed in Pymunk. Dataset
> actions are used only to define the recorded demonstration, never to move the
> evaluation rollout.

> **SUCCESS GATE 5 — OBJECT POSE AND HOLD.** Both block position and wrapped
> angle must satisfy the threshold for the full hold duration. The final pusher
> location cannot create either a success or a failure.

> **AUDIT GATE 6 — FIXED IDENTITY.** Model and random baselines must share the
> manifest SHA-256, dataset fingerprint, protocol, policy seed, solver budget,
> and environment fingerprint.

Block speed below `5 px/s` is reported as an optional settled-placement
diagnostic. It is not silently folded into primary SR.

## Benchmark success versus LeWM planning cost

> **INTERFACE BOUNDARY — BENCHMARK VS. PLANNER.** CLEAR fixes the evaluated
> start-goal distribution and decides whether the executed rollout completes
> the task. It does not prescribe, replace, or post-process an algorithm's
> internal planning objective.

The canonical LeWM/CEM planner still ranks candidate actions with the latent
distance to the complete goal image, schematically

```text
J_latent = ||z_predicted - z_goal||^2.
```

Because the goal RGB contains both the block and the pusher, this surrogate can
assign nonzero cost to a task-complete state whose pusher stops somewhere other
than in the demonstration. It can also partially reward matching the pusher
while the block remains incorrect. That is a potential semantic ambiguity in
the planner's cost, not a reason to make pusher pose part of benchmark success.

CLEAR therefore leaves the LeWM cost untouched and reports success from the
executed block trajectory. In the audited setting below, the original
full-image latent cost still reaches `93%` Moderate and `79%` Strict SR, so the
ambiguity is not catastrophic for the tested LeWM/CEM configuration. The
benchmark simply prevents latent similarity from being mistaken for task
completion.

## Why the released predicate is not the task

Historical success computes a single position norm over the first four state
coordinates. It can reject a correctly placed block because the pusher stops
elsewhere. Historical pair construction can also treat a large pusher move as
task difficulty even when the T block barely moves. CLEAR fixes both sides:
pair difficulty and rollout success use the same object semantics.

## Audited result

Official high-epoch LeWM, 100 fixed seed-42 pairs, `300 x 30` CEM, solver batch
size 1:

| Mode | LeWM | Random | Excess over random |
|---|---:|---:|---:|
| Moderate | **93%** | 7% | +86 pp |
| Strict | **79%** | 2% | +77 pp |

![PushT task-semantic trace](../../assets/task_gifs/pusht.gif)

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.3/pusht/strict-seed42-n100.json \
  --policy official/pusht --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/pusht_expert_train.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --output results/pusht-strict.json
```

Use the paired random output from
[`results/v0.3/`](../../results/v0.3/) when reporting excess over random.
