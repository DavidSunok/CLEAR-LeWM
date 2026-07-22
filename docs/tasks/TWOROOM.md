# TwoRoom evaluation guide

## What the task should measure

When start and goal lie in different rooms, the complete circular agent must
pass through a door with sufficient clearance. Crossing a solid wall is never
a valid solution.

The released expert follows this interpretation: it moves to the nearest door
that fits, then to the goal. However, the target constraint intended to enforce
cross-room difficulty is commented out in the released environment, so mixed
manifests contain many direct same-room goals.

## Released predicate and failure mode

The released collision path can accept a motion whose endpoint is clear even
when the segment crosses a wall or clips a doorframe. Endpoint-only success
then grants credit to an invalid route.

## CLEAR v0.3 correction

- all headline pairs are cross-room and initially full-disk clear;
- every action segment uses swept-circle continuous collision;
- door intervals are eroded by the complete agent radius;
- a cross-room success requires a valid door crossing;
- `route_valid=false` makes endpoint proximity insufficient.

| Mode | Goal distance | Hold | Minimum valid-door path |
|---|---:|---:|---:|
| Moderate | `<12 px` | 3 steps | 24 px |
| Strict | `<8 px` | 5 steps | 32 px |

## Audited result

| Mode | LeWM | Random | Invalid routes after correction |
|---|---:|---:|---:|
| Moderate | **61%** | 2% | 0 / 100 |
| Strict | **24%** | 0% | 0 / 100 |

On the same 100 Strict cross-room goals, original dynamics produced 37%
endpoint SR, but only 6% of those unchanged trajectories survived route
gating. Swept collision recovered 24% legal SR while removing all invalid
routes.

![TwoRoom continuous collision comparison](../../assets/task_gifs/tworoom.gif)

[Watch the dedicated 1080p topology video](../../assets/showcase/tworoom_topology_1080p.mp4).

## Run it

```bash
clear-lewm evaluate \
  --manifest manifests/v0.3/tworoom/strict-seed42-n100.json \
  --policy official/tworoom --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/tworoom.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --output results/tworoom-strict.json
```

The manifest embeds the swept and route-required protocol. No separate runtime
patch is needed.
