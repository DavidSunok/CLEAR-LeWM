# CLEAR-LeWM Data Specification v0.3

This document separates a **logical dataset snapshot** from its physical storage
format. HDF5, Lance, and a derived memory map are interchangeable only after the
equivalence checks below pass; a filename match is not sufficient.

## Canonical four-task snapshots

| Task | Canonical logical dataset | Rows | Episodes | Current joint-training reader |
|---|---|---:|---:|---|
| PushT | `pusht_expert_train` | 2,336,736 | 18,685 | Lance |
| Cube | `ogbench/cube_single_expert` | 2,010,000 | 10,000 | HDF5 |
| Reacher | `dmc/reacher_random` | 2,010,000 | 10,000 | HDF5 |
| TwoRoom | `tworoom` | 920,809 | 10,000 | HDF5 |

The four-task shared-encoder experiment therefore uses one Lance reader
(PushT) and three HDF5 readers. The local `reacher.h5` and
`dmc/reacher_random.h5` names resolve to the same file snapshot; published
records should use the canonical `dmc/reacher_random` name.

Evaluation manifests are always generated from the canonical HDF5 snapshot.
This gives one auditable row index, episode ID, start step, goal step, and
metadata fingerprint regardless of the format used during training.

## Format-equivalence contract

A derived format may replace the canonical reader for training only when an
audit verifies all of the following:

1. identical episode counts, episode lengths, and episode offsets;
2. identical sample and clip counts for the declared `num_steps` and
   `frameskip`;
3. identical action ordering and action-chunk reshape semantics;
4. identical non-image columns, or declared numeric tolerances caused only by
   serialization precision;
5. decoded image tensors compared on fixed boundary rows and a seeded random
   sample, with mean absolute error and maximum error reported;
6. transforms run before action chunks are reshaped, matching the LeWM dataset
   base-class contract;
7. normalization statistics computed from the same canonical full-data scope;
8. the source revision, conversion command, schema, and audit report retained.

The current PushT audit found equal row counts and action statistics between
HDF5 and Lance. Sampled state/proprio means differed at roughly `1e-4`; decoded
pixels had MAE `0.108/255` and maximum absolute error `19/255`. These formats
are close enough to motivate an empirical training-parity ablation, but they
are not byte-identical and must not be described that way.

## FAST memory-map reader

`clear_lewm.fast_dataset.FastMemmapDataset` is a derived I/O path, not a new
dataset. It stores already-decoded tensors in raw memory maps and preserves the
episode and clip semantics of the source reader. Four named profiles pin the
same source and schema used by joint training:

| Profile | Authoritative source | Numeric FAST columns |
|---|---|---|
| `pusht` | `pusht_expert_train.lance` | pixels, action, proprio, state |
| `cube` | `ogbench/cube_single_expert.h5` | pixels, action, observation, merged proprio |
| `reacher` | `dmc/reacher_random.h5` | pixels, action, observation |
| `tworoom` | `tworoom.h5` | pixels, action, proprio |

Convert, audit, and benchmark any profile with the same interface:

```bash
python scripts/preprocess_fast_dataset.py \
  --task pusht \
  --cache-dir "$STABLEWM_HOME" \
  --output "$STABLEWM_HOME/preprocessed/pusht"

python scripts/audit_fast_dataset.py \
  --task pusht \
  --cache-dir "$STABLEWM_HOME" \
  --fast-dir "$STABLEWM_HOME/preprocessed/pusht"

python scripts/benchmark_fast_dataset.py \
  --task pusht \
  --cache-dir "$STABLEWM_HOME" \
  --fast-dir "$STABLEWM_HOME/preprocessed/pusht"
```

Replace `pusht` with `cube`, `reacher`, or `tworoom` for the other training
inputs. The 2026-07-23 audit checked 258 fixed and seeded clips per task, full
episode metadata, and every non-image column. All four snapshots matched the
authoritative readers exactly, including Cube's 19-dimensional merged proprio
and complete unstrided action chunks. A FAST artifact is not eligible for a
reported run until this audit passes.

## Required training data record

Every reported model should retain, per task:

- canonical logical dataset name and immutable revision or fingerprint;
- physical reader (`hdf5`, `lance`, or `fast-memmap`);
- row count, episode count, `num_steps`, and `frameskip`;
- exact train/held-out episode IDs;
- normalization scope and statistics fingerprint;
- conversion and equivalence-audit report for derived formats.

Storage-format ablations must change only the reader. Model initialization,
batch order, optimizer steps, transforms, and evaluation manifest stay fixed.
