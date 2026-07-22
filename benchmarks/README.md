# Performance records

These machine-readable records support the measurements in
[`PERFORMANCE.md`](../PERFORMANCE.md).

- `fast_audit_{pusht,cube,reacher,tworoom}_20260723.json`: exact
  source-equivalence audits for all four named FAST training profiles.
- `fast_loader_{pusht,cube,reacher,tworoom}_20260723.json`: five-pair,
  steady-state Source/FAST loader throughput after an untimed priming pair.
- `fast_audit_pusht_20260721.json`: earlier corrected PushT snapshot audit.
- `fast_loader_pusht_20260721.json`: earlier H20-3 Lance/FAST throughput.
- `cem_batch_pusht_20260721.json`: official and throughput-mode CEM batch sweep.
- `cem_batch_four_task_20260722.json`: fixed-pair four-task batch-1/batch-16
  trajectory-equivalence audit for a LeWM-compatible custom checkpoint.

FAST records are training-input audits, not evaluation results. Only the CEM
batch-1 row is reference-equivalent. Performance records are not part of
`results/reference/` and must not be used to replace protocol SR tables.
