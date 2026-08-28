# Submitting results

CLEAR-LeWM accepts community methods through compact, auditable pull requests.
The repository stores result records, hashes, and method metadata. It does not
store training datasets, model weights, or generated videos.

## What may be submitted

| Contribution | Entry point | Comparison status |
|---|---|---|
| A method on fixed v0.5 Moderate manifests | result bundle PR | eligible for the Moderate table |
| A method on fixed v0.5 Strict manifests | result bundle PR | eligible for the Strict table |
| A LeWM-compatible policy adapter | proposal issue, then code PR | eligible after tests and review |
| Reduced or external training data | proposal issue, then a separate data-track PR | never mixed silently with standard-data results |
| A success-rule or manifest change | protocol proposal issue | requires a new benchmark version |
| A new task | protocol proposal issue | separate experimental track until calibrated |

Raw datasets and weights should live in durable external storage such as a
public project release or dataset/model hub. Every external artifact must have
a stable URL, revision, license, and SHA-256 where applicable.

## Verification levels

| Badge | Meaning |
|---|---|
| `self-reported` | JSON structure, canonical manifest, trace arithmetic, hashes, and provenance pass CI |
| `reproducible` | the submission also provides public code, a checkpoint, and an exact evaluation command |
| `maintainer-verified` | a CLEAR-LeWM maintainer independently reran the submitted revision and matched the declared result within the protocol's deterministic contract |

Authors may request the first two levels. Only maintainers assign
`maintainer-verified` after reproduction.

## Fixed comparison contract

v0.5 submissions must use:

- the checked-in `v0.5` manifest for the declared Moderate or Strict mode;
- all 100 selected episodes and the manifest policy seed;
- the exact task predicate embedded in that manifest;
- the canonical paired random output from `results/v0.5/`;
- solver batch size 1 when a sampling solver is used;
- a full `clear-lewm-result-v1` record with episode outcomes and runtime
  fingerprints.

Methods may use CEM, another planner, or direct inference. Their inference mode,
solver budget, runtime, and training-data track remain visible in the record;
unlike compute budgets must not be presented as matched-compute comparisons.

## Submission layout

Create one immutable bundle per method revision:

```text
submissions/<github-user>/<method>-<revision>/
├── METHOD_CARD.md
├── submission.json
└── results/
    └── pusht-moderate.json
```

Start from [`submissions/submission.example.json`](../submissions/submission.example.json).
The method card should describe architecture, training data, checkpoint source,
inference mode, compute budget, known limitations, and the exact commands used.

Generate each result with the canonical manifest, for example:

```bash
clear-lewm evaluate \
  --manifest manifests/v0.5/pusht/moderate-seed42-n100.json \
  --policy /path/to/policy \
  --policy-label your-method \
  --random-results results/v0.5/pusht-moderate-random-seed42-n100.json \
  --solver-batch-size 1 \
  --output submissions/USER/METHOD/results/pusht-moderate.json
```

Record the immutable result digest:

```bash
sha256sum submissions/USER/METHOD/results/*.json
```

Then validate the bundle before opening a PR:

```bash
clear-lewm validate-submission submissions/USER/METHOD/submission.json
python scripts/build_community_leaderboard.py
ruff check .
pytest
```

The leaderboard command refreshes the generated website registry, compact
README table, and matched comparison figure. Commit
`submissions/leaderboard.json`, `assets/community_model_comparison.png`, and
`README.md` with the bundle so an accepted method appears on the public
homepage immediately.

The validator recomputes result SR from all episode outcomes and rejects
noncanonical manifests, altered protocols, evaluation-dataset drift, incorrect
paired-random values, invalid TwoRoom routes, missing runtime fingerprints, and
result files whose SHA-256 does not match the submission record.

## Pull-request review

1. Fork `DavidSunok/CLEAR-LeWM` and create a method-specific branch.
2. Add the bundle under `submissions/`, then run the leaderboard builder and
   include its generated registry/README updates. Add an adapter or tests only
   when required by the method.
3. Open a PR using the repository template and disclose every training-data and
   inference difference.
4. Let CI validate the result. A passing CI run establishes structural
   validity, not scientific endorsement.
5. Address review. Accepted results retain their original JSON and revision;
   reruns are new bundles rather than silent replacements.

For a new training dataset, task, environment, or success rule, open the
`Benchmark or data-track proposal` issue first. Include task semantics,
licensing, data provenance, random and no-op floors, calibration evidence, and
a migration plan. A protocol change never rewrites an existing version.

## Privacy and licensing

Everything in a submission PR becomes public. Do not include credentials,
private repositories, unpublished private checkpoints, personal dataset paths,
or artifacts without redistribution permission. The method remains under its
own stated license; contributions to CLEAR-LeWM are accepted under this
repository's MIT license.
