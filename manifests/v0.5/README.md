# CLEAR-LeWM v0.5 canonical manifests

v0.5 contains two complementary protocols for all four tasks and three fixed
evaluation seeds:

```text
manifests/v0.5/<task>/<moderate|strict>-seed<0|1|42>-n100.json
```

**Moderate** minimally repairs the released evaluation: pre-solved pairs are
removed, Reacher joint topology is corrected, and TwoRoom uses clean source
windows plus swept-disk physics. PushT and Cube retain the released predicates.

**Strict** inherits those corrections and applies tighter task semantics:
T-block-only PushT, symmetry-aware cube pose, fingertip-endpoint Reacher, and a
legal cross-room route with a tighter endpoint gate.

## Seed-42 manifest audit

| Task | Mode | Valid `+25` pairs | Pairs after filters | Eligible episodes | Selected |
|---|---|---:|---:|---:|---:|
| PushT | Moderate | 1,869,611 | 1,865,441 | 18,685 | 100 |
| PushT | Strict | 1,869,611 | 1,440,427 | 18,567 | 100 |
| Cube | Moderate | 1,760,000 | 1,084,570 | 10,000 | 100 |
| Cube | Strict | 1,760,000 | 1,129,326 | 10,000 | 100 |
| Reacher | Moderate | 1,760,000 | 1,750,029 | 10,000 | 100 |
| Reacher | Strict | 1,760,000 | 1,720,542 | 10,000 | 100 |
| TwoRoom | Moderate | 670,809 | 8,294 | 1,415 | 100 |
| TwoRoom | Strict | 670,809 | 8,325 | 1,421 | 100 |

Every selected pair is initially unsolved under its own mode. Every TwoRoom
pair is cross-room, endpoint-clear, and clean across all 25 source transitions.
The full normative contract is in [`../../EVALUATION_SPEC.md`](../../EVALUATION_SPEC.md).

Earlier public suites are preserved by their Git release tags; their manifests
are not relabeled as v0.5 artifacts.
