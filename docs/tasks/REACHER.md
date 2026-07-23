# Reacher evaluation guide

> **Task contract:** drive every periodic target joint into the desired
> configuration. Primary SR measures first valid arrival; maintaining the pose
> is a separate stabilization claim.

[Back to the four task contracts](../../README.md#what-v03-fixes) ·
[Normative v0.3 specification](../../EVALUATION_SPEC.md)

## What the policy sees and what the evaluator does

1. A manifest selects a start and a goal exactly 25 environment steps later
   from one DMC Reacher episode.
2. The DMC simulator is reset to the recorded start and the future RGB frame is
   supplied as the goal image.
3. The evaluated policy receives pixels and emits its own actions; expert
   actions are not replayed.
4. DMC advances the arm under those actions.
5. The evaluator reads joint coordinates after each physical step and computes
   the shortest periodic error. Joint state is not exposed to the policy.

## Official and CLEAR manifest construction

| Manifest decision | Historical Official | CLEAR Moderate | CLEAR Strict |
|---|---|---|---|
| Pair source | same episode, `goal = start + 25` | same | same |
| Sampling unit | rows | episodes | episodes |
| Initially solved pair | retained | removed with wrapped `<0.075 rad` predicate | removed with wrapped `<0.05 rad` predicate |
| Joint geometry | raw subtraction | shortest periodic difference | shortest periodic difference |
| Minimum wrapped motion | none | none beyond the unsolved gate | `>=0.25 rad` in at least one joint |

> **MANIFEST GATE 1 — SAME-EPISODE FUTURE.** Every target is tied to a real
> future configuration in the selected episode.

> **MANIFEST GATE 2 — NOT PRE-SOLVED.** The start must fail the same wrapped
> joint predicate used during rollout.

> **MANIFEST GATE 3 — STRICT JOINT MOTION.** Strict requires a maximum wrapped
> start-goal joint displacement of at least `0.25 rad`.

The released seed-42 manifests contain 100 unique episodes per mode and zero
initial successes. Strict's minimum observed maximum wrapped displacement is
`0.2525 rad`.

## Official and CLEAR runtime success

For one periodic coordinate, CLEAR uses

```text
abs(atan2(sin(q - q_goal), cos(q - q_goal))).
```

| Runtime decision | Historical Official | CLEAR Moderate | CLEAR Strict |
|---|---|---|---|
| Joint error | raw coordinate subtraction | maximum wrapped error | maximum wrapped error |
| Threshold | `<0.05 rad` | `<0.075 rad` | `<0.05 rad` |
| Primary temporal rule | first hit | first hit | first hit |
| Stabilization | not separated | report `SR@hold2` separately | report `SR@hold2` separately |

> **ROLLOUT GATE 4 — PHYSICAL ACTION EXECUTION.** The policy controls the DMC
> arm in closed loop. The evaluator does not score a predicted state or replay
> an expert trajectory.

> **SUCCESS GATE 5 — WRAPPED FIRST ARRIVAL.** Every target joint must be within
> the mode threshold at the same physical step. Crossing `-pi/pi` cannot create
> a false failure.

> **REPORTING GATE 6 — ARRIVAL IS NOT HOLDING.** First-hit SR is the primary
> released task meaning. `SR@hold2` and terminal joint speed must remain
> separately labeled diagnostics.

> **AUDIT GATE 7 — FIXED IDENTITY.** Model and random must share manifest hash,
> dataset fingerprint, protocol, solver budget, and seeds.

## Why wrapping and temporal separation both matter

Raw subtraction can treat nearby configurations on opposite sides of
`-pi/pi` as far apart. Conversely, silently adding a hold changes an arrival
task into a stabilization task. CLEAR repairs the geometry while preserving
the released first-hit claim, then reports holding separately.

## Audited result

| Mode | LeWM | Random | Excess over random |
|---|---:|---:|---:|
| Moderate | **90%** | 17% | +73 pp |
| Strict | **36%** | 4% | +32 pp |

On Strict pairs, adding a two-step hold changes the result to `1% / 0%`. That
number is a stabilization diagnostic, not a replacement for first-hit SR.

![Reacher wrapped-angle trace](../../assets/task_gifs/reacher.gif)

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.3/reacher/strict-seed42-n100.json \
  --policy official/reacher --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/reacher.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --output results/reacher-strict.json
```
