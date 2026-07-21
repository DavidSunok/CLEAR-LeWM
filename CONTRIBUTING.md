# Contributing

Thank you for helping make latent world-model evaluation easier to audit.

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

Do not commit datasets, model checkpoints, videos, credentials, private code,
or machine-specific absolute paths. Public reference manifests and compact JSON
results are welcome when they include immutable dataset and code revisions.

## Attribution

Do not remove upstream LeWM or stable-worldmodel attribution. Changes under the
LeWM submodule should be proposed upstream or implemented as an explicit adapter
in CLEAR-LeWM.
