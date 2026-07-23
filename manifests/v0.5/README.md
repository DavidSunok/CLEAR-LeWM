# CLEAR-LeWM v0.5 Moderate manifests

These four immutable manifests define the canonical seed-42, 100-episode v0.5
Moderate evaluation:

```text
manifests/v0.5/<task>/moderate-seed42-n100.json
```

Every task uses a same-episode goal exactly 25 steps after the start,
episode-balanced sampling, the full public-data split, and zero initially
successful selected pairs. No minimum-displacement or hold rule is added.

| Task | Valid `+25` pairs | Pairs after v0.5 filters | Selected episodes |
|---|---:|---:|---:|
| PushT | 1,869,611 | 1,865,441 | 100 |
| Cube | 1,760,000 | 1,084,570 | 100 |
| Reacher | 1,760,000 | 1,750,029 | 100 |
| TwoRoom | 670,809 | 8,294 | 100 |

TwoRoom's final pool additionally requires cross-room endpoints, full-disk
endpoint clearance, and 25/25 legal source transitions. Every selected
TwoRoom pair records `source_window_clean=true`.

Do not edit a manifest after a result cites its SHA-256. Protocol changes
require a new benchmark version.
