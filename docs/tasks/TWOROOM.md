# TwoRoom evaluation guide

> **v0.5 Moderate contract:** evaluate cross-room navigation with corrected
> swept-disk physics and the released endpoint success threshold.

[Back to the task contracts](../../README.md#what-v05-fixes) | [Normative
v0.5 specification](../../EVALUATION_SPEC.md)

## What is corrected

The canonical PLDM/DINO-WM task samples start and goal in opposite rooms. The
released LeWM/stable-worldmodel rewrite checks collision at the requested
endpoint, which can admit transitions whose segment crosses a solid wall or
clips a doorway. v0.5 resolves every runtime move as a complete radius-7 disk
against the wall continuously.

The released offline dataset also contains transitions produced by that defect.
Moderate therefore accepts a start-goal pair only when every transition in its
25-step source window is legal under the corrected geometry. This is a data
quality gate, not expert-action replay.

## Gates

| Gate | v0.5 Moderate definition |
|---|---|
| Pair source | same episode, exact `+25` step future |
| Sampling | episode-balanced, cross-room only |
| Endpoint geometry | complete start and goal disks clear |
| Source window | all 25 recorded transitions swept-disk legal |
| Runtime collision | continuous swept disk with full-radius door clearance |
| Runtime success | endpoint distance `<16 px` |
| Temporal rule | first hit; no hold |
| Route condition | diagnostic only; not a second success predicate |

Cross-room is a pair-construction rule. Once evaluation starts, success remains
the official endpoint threshold because corrected collision already makes wall
penetration impossible. The result still stores wall contacts, blocked steps,
valid crossings, positions, and route-valid diagnostics.

## Canonical v0.5 manifest audit

The released dataset contains 670,809 valid `+25` pairs. Exactly 8,294 pairs
from 1,415 episodes remain after initial-success, cross-room, endpoint-clear,
and source-window-clean filtering. The fixed seed-42 manifest selects 100
distinct episodes; all 100 have `source_window_clean=true`.

Official LeWM reaches **81%** on the fixed v0.5 Moderate manifest; paired
random reaches 6%, with corrected swept-disk execution.

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.5/tworoom/moderate-seed42-n100.json \
  --policy official/tworoom/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/tworoom.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/v0.5/tworoom-moderate-random-seed42-n100.json \
  --output results/tworoom-v05-moderate.json
```
