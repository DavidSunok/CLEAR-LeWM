# Reference manifests

The primary v0.2 suite contains four tasks, three tiers, 100 pairs per manifest,
and manifest/policy seed 42:

```text
manifests/{pusht,cube,reacher,tworoom}/{official,moderate,strict}-seed42-n100.json
```

`official` reproduces upstream `rng.choice(len(valid_indices) - 1, ...)`
row sampling. `moderate` and `strict` use 100 episode-balanced pairs and remove
every pair already satisfying that tier's success geometry. Strict also applies
the task-specific minimum displacement in `EVALUATION_SPEC.md`.

The v0.1 `official-compat`, `clear-id`, `clear-standard`, and `clear-hard`
manifests remain checked in for reproducibility. They are legacy artifacts and
are not aliases for the new three-tier manifests.

Each JSON embeds protocol parameters and a dataset metadata fingerprint. Final
archival results should also cite the immutable public dataset revision or full
dataset SHA-256. Released full-data checkpoints are in-distribution even when a
manifest happens to select the deterministic held-out episode IDs.
