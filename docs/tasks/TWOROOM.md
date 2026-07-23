# TwoRoom evaluation guide

> **Moderate repairs the rewritten environment while preserving endpoint
> success. Strict additionally requires a precise, legal cross-room route.**

[Back to the task contracts](../../README.md#two-auditable-modes) | [Normative
v0.5 specification](../../EVALUATION_SPEC.md)

## What Moderate repairs

The PLDM/DINO-WM TwoRoom task places start and goal in opposite rooms. The
released LeWM/stable-worldmodel rewrite checks collision at the requested
endpoint, which can admit a segment through a solid wall or a radius-7 disk
clipping a doorway. v0.5 continuously sweeps the full disk along every move.

The released offline dataset also contains transitions produced by the faulty
rewrite. A pair is eligible only when all 25 transitions in its source window
are legal under corrected geometry. This is a data-quality gate; evaluation
still executes only the tested policy's actions.

## Gates

| Gate | v0.5 Moderate | v0.5 Strict |
|---|---|---|
| Pair | cross-room, clear endpoints, 25/25 clean source transitions | same |
| Runtime physics | continuous swept disk with full-radius doorway clearance | same |
| Endpoint | distance `< 16 px` | distance `< 8 px` |
| Route | recorded diagnostic | `route_valid=true` and at least one legal crossing |
| Goal side | implied by endpoint in most cases | explicitly required |
| Temporal rule | first hit | first hit after all gates pass |

`route_valid=true` means no executed segment crosses solid wall geometry. A
valid room crossing is counted only when the complete disk moves through a
doorway from one room side to the other. Strict also verifies that the final
agent lies on the goal side of the wall.

## Canonical dataset audit

The released dataset has 670,809 valid `+25` pairs. After Moderate filters,
8,294 pairs from 1,415 episodes remain; Strict retains 8,325 pairs from 1,421
episodes because its tighter endpoint rule marks fewer starts as pre-solved.

## Official reference

- Moderate seed 42: official LeWM **81%**, paired random **6%**.
- Strict seeds 0/1/42: official LeWM **61/57/57%**; random **5/0/0%**.
- All checked-in Strict rollouts report zero invalid routes.

## Reproduce

```bash
clear-lewm evaluate \
  --manifest manifests/v0.5/tworoom/strict-seed42-n100.json \
  --policy official/tworoom/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/tworoom.h5 \
  --num-samples 300 --n-steps 30 --topk 30 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/v0.5/tworoom-strict-random-seed42-n100.json \
  --output results/tworoom-v05-strict.json
```
