# PushT evaluation guide

## What the task should measure

PushT asks the controller to place the T-shaped block at a target planar pose.
The pusher is an instrument, not part of the completed object state.

## Released predicate and failure mode

The released predicate computes position error over the first four state
coordinates, which concatenate pusher position and block position. A block can
be correctly placed yet rejected because the pusher finishes elsewhere. The
same coordinates also made pusher travel look like task difficulty.

## CLEAR v0.3 correction

CLEAR uses block position `state[2:4]`, wrapped block angle, and block
translation for pair difficulty. Every selected pair is initially unsolved
under the same predicate used during rollout.

| Mode | Block position | Block angle | Hold | Minimum block translation |
|---|---:|---:|---:|---:|
| Moderate | `<20 px` | `<20 deg` | 3 steps | 25 px |
| Strict | `<15 px` | `<15 deg` | 5 steps | 50 px |

Block speed below `5 px/s` is a useful settled-placement diagnostic, but it is
not silently folded into primary SR.

## Audited result

Official high-epoch LeWM, 100 fixed pairs, seed 42, `300 x 30` CEM, batch 1:

| Mode | LeWM | Random | Excess over random |
|---|---:|---:|---:|
| Moderate | **93%** | 7% | +86 pp |
| Strict | **79%** | 2% | +77 pp |

![PushT task-semantic trace](../../assets/task_gifs/pusht.gif)

## Run it

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

