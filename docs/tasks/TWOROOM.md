# TwoRoom evaluation guide

> **Task contract:** move the complete circular agent from one room to the
> target in the other room through a door with full-body clearance. Crossing or
> clipping a solid wall is never a valid solution.

[Back to the four task contracts](../../README.md#what-v03-fixes) ·
[Normative v0.3 specification](../../EVALUATION_SPEC.md)

## What the policy sees and what the evaluator does

1. A manifest selects a start and a goal exactly 25 environment steps later
   from the same navigation episode.
2. The simulator is reset to the start position and the full future RGB frame
   is supplied as the goal image.
3. The evaluated policy receives pixels and emits its own actions; expert
   actions are not replayed.
4. Every requested motion is resolved with swept-disk collision before the
   simulator accepts the next agent position.
5. The evaluator records every segment, wall contact, valid door crossing,
   route-valid bit, endpoint distance, and hold count. Privileged geometry is
   used only by the evaluator.

## Official and CLEAR manifest construction

| Manifest decision | Historical Official | CLEAR Moderate | CLEAR Strict |
|---|---|---|---|
| Pair source | same episode, `goal = start + 25` | same | same |
| Sampling unit | rows | episodes | episodes |
| Initially solved pair | retained | removed under `<12 px` endpoint rule | removed under `<8 px` endpoint rule |
| Room relation | same-room and cross-room mixed | cross-room only | cross-room only |
| Start/goal clearance | not gated by full agent disk | both full-disk clear | both full-disk clear |
| Minimum valid-door path | none | `>=24 px` | `>=32 px` |

> **MANIFEST GATE 1 — SAME-EPISODE FUTURE.** Every goal remains a real future
> state from the selected demonstration.

> **MANIFEST GATE 2 — CROSS-ROOM TASK.** Start and goal must lie in different
> rooms; a direct same-room move cannot enter the headline set.

> **MANIFEST GATE 3 — FULL-DISK CLEARANCE.** The complete agent disk must fit at
> both endpoints, and the shortest valid-door path must exceed the mode's
> difficulty threshold.

The released seed-42 manifests contain 100 unique episodes per mode, zero
initial successes, and 100/100 cross-room, start-clear, and goal-clear pairs.
The minimum observed valid-door path is `32.472 px`.

## Official and CLEAR runtime success

| Runtime decision | Historical Official | CLEAR Moderate | CLEAR Strict |
|---|---|---|---|
| Collision | released endpoint update can cross or clip walls | swept complete disk | swept complete disk |
| Door clearance | endpoint only | door interval eroded by agent radius | same |
| Valid crossing required | no | yes | yes |
| Goal distance | `<16 px` | `<12 px` | `<8 px` |
| Temporal rule | first hit | 3 consecutive valid steps | 5 consecutive valid steps |

> **ROLLOUT GATE 4 — CONTINUOUS COLLISION.** Every action segment is checked as
> a swept disk. A clear endpoint cannot excuse a path that crossed a wall.

> **ROUTE GATE 5 — VALID TOPOLOGY.** `route_valid` must remain true and a
> cross-room episode must record at least one full-clearance door crossing.

> **SUCCESS GATE 6 — LEGAL ENDPOINT AND HOLD.** Endpoint proximity counts only
> after the route gates pass, and must persist for the complete hold duration.

> **AUDIT GATE 7 — TRACEABLE PHYSICS.** Every result stores invalid-route count,
> collision contacts, blocked steps, valid crossings, positions, manifest hash,
> and environment fingerprint.

## Why endpoint success is insufficient

The released collision path can accept a motion whose endpoint is clear even
when the segment crossed a wall or clipped a doorframe. The released target
constraint also permits many same-room pairs. CLEAR makes the topology part of
both pair construction and runtime success, so an endpoint cannot launder an
invalid trajectory.

## Audited result

| Mode | LeWM | Random | Invalid routes after correction |
|---|---:|---:|---:|
| Moderate | **61%** | 2% | 0 / 100 |
| Strict | **24%** | 0% | 0 / 100 |

On the same 100 Strict cross-room goals, original dynamics produced 37%
endpoint SR, but only 6% of the unchanged trajectories survived route gating.
Swept collision recovered 24% legal SR while removing all invalid routes.

![TwoRoom continuous collision comparison](../../assets/task_gifs/tworoom.gif)

[Watch the dedicated 1080p topology video](../../assets/showcase/tworoom_topology_1080p.mp4).

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.3/tworoom/strict-seed42-n100.json \
  --policy official/tworoom --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/tworoom.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --output results/tworoom-strict.json
```

The manifest embeds the swept and route-required protocol. CLEAR installs the
corresponding runtime gate automatically.
