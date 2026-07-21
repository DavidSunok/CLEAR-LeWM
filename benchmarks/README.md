# Performance records

These machine-readable records support the measurements in
[`PERFORMANCE.md`](../PERFORMANCE.md).

- `fast_audit_pusht_20260721.json`: source-equivalence audit for the corrected
  PushT FAST snapshot.
- `fast_loader_pusht_20260721.json`: isolated Lance/FAST loader throughput.
- `cem_batch_pusht_20260721.json`: official and throughput-mode CEM batch sweep.
- `cem_batch_four_task_20260722.json`: fixed-pair four-task batch-1/batch-16
  trajectory-equivalence audit for a LeWM-compatible custom checkpoint.

Only the CEM batch-1 row is reference-equivalent. Performance records are not
part of `results/reference/` and must not be used to replace protocol SR tables.
