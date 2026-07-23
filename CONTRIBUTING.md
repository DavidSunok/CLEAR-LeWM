# Contributing

Thank you for helping make latent world-model evaluation easier to audit.

## Choose a contribution path

- **Fixed-protocol method result:** follow
  [`docs/SUBMITTING_RESULTS.md`](docs/SUBMITTING_RESULTS.md), validate the bundle,
  and open a PR.
- **New adapter:** open the `Result submission or method integration` issue
  before implementing a runner integration.
- **New dataset, task, metric, manifest, or success rule:** open the
  `Benchmark or data-track proposal` issue first. These changes require
  calibration and explicit versioning.
- **Bug, test, documentation, or performance improvement:** open an issue or a
  focused PR with reproduction evidence.

Community results are labeled `self-reported`, `reproducible`, or
`maintainer-verified`. Passing CI establishes internal consistency; only an
independent maintainer rerun earns the final label.

Do not upload a training dataset or checkpoint into Git. Store large artifacts
in durable public storage and submit their URL, immutable revision, license,
and digest. Standard, reduced-data, and external-data training runs remain
separate tracks even though they use the same fixed evaluation manifests.

## Protocol changes

A pull request that changes sampling, splitting, success criteria, or metric
aggregation must:

1. update `EVALUATION_SPEC.md` and increment the relevant schema version;
2. add a synthetic regression test demonstrating the intended behavior;
3. preserve old protocol names and behavior;
4. include a migration note explaining whether old and new numbers are
   comparable.

## Code changes

```bash
pip install -e '.[dev]'
ruff check .
pytest
```

Runner changes must preserve strict checkpoint loading, custom-runtime target
provenance, atomic result writes, and physics/numerical environment records.
Evaluation changes should include one full-path smoke in the pinned reference
environment in addition to synthetic unit tests.

Do not commit datasets, model checkpoints, submission videos, credentials,
private code, or machine-specific absolute paths. Maintainer-generated website
media, public reference manifests, and compact JSON results are welcome when
they include immutable dataset and code revisions.

## Community result changes

A result PR must contain one immutable bundle below `submissions/`, including a
method card, `submission.json`, and the generated result JSON files. Run:

```bash
clear-lewm validate-submission submissions/USER/METHOD/submission.json
ruff check .
pytest
```

The validator checks canonical manifests, protocols, policy seeds, evaluation
data identity, episode traces, result hashes, paired-random arithmetic,
runtime fingerprints, checkpoint provenance, and TwoRoom route legality.

## Attribution

Do not remove upstream LeWM or stable-worldmodel attribution. Changes under the
LeWM submodule should be proposed upstream or implemented as an explicit adapter
in CLEAR-LeWM.
