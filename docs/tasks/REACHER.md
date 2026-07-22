# Reacher evaluation guide

## What the task should measure

The released `qpos_match` task terminates when every target joint is reached.
It is a first-hit task; stable holding is a stronger control claim.

## Released predicate and failure mode

Joint coordinates are periodic, but raw subtraction can treat two nearby
angles on opposite sides of `-pi/pi` as far apart. Earlier robust protocols
also mixed a looser geometric threshold with a multi-step hold, obscuring
whether a score meant arrival or stabilization.

## CLEAR v0.3 correction

CLEAR computes the shortest wrapped angle per joint. Primary SR preserves
first-hit semantics; `SR@hold2` and terminal joint speed are separate
diagnostics.

| Mode | Maximum wrapped joint error | Temporal rule |
|---|---:|---|
| Moderate | `<0.075 rad` | first hit |
| Strict | `<0.05 rad` | first hit |

## Audited result

| Mode | LeWM | Random | Excess over random |
|---|---:|---:|---:|
| Moderate | **90%** | 17% | +73 pp |
| Strict | **36%** | 4% | +32 pp |

On the Strict pairs, adding a two-step hold changes the result to `1% / 0%`.
That number is a stabilization diagnostic, not a replacement for first-hit SR.

![Reacher wrapped-angle trace](../../assets/task_gifs/reacher.gif)

## Run it

```bash
clear-lewm evaluate \
  --manifest manifests/v0.3/reacher/strict-seed42-n100.json \
  --policy official/reacher --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/reacher.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --output results/reacher-strict.json
```

