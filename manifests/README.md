# Reference manifests

This directory contains the first versioned CLEAR-LeWM evaluation manifests:

- four tasks: PushT, Reacher, TwoRoom, and OGBench-Cube;
- four tracks: `official-compat`, `clear-id`, `clear-standard`, and `clear-hard`;
- 100 pairs per manifest;
- manifest and policy seed 42.

`official-compat` start rows exactly reproduce the upstream
`rng.choice(len(valid_indices) - 1, ...)` behavior for `num_eval=100`.

All three CLEAR tracks contain 100 distinct episode IDs and no pair that meets
the task's success threshold at the initial state. The JSON files include the
dataset metadata fingerprint used during generation. A final archival release
should additionally publish full dataset SHA256 values or immutable dataset
revision identifiers.

Released full-data checkpoints may use `official-compat` and `clear-id`.
`clear-standard` and `clear-hard` support held-out claims only when their
episode split was excluded from model training.
