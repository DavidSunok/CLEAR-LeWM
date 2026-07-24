# Community submissions

This directory is the public registry for community evaluation bundles. Each
accepted bundle is immutable and validated against canonical CLEAR-LeWM
manifests by CI.

```text
submissions/<github-user>/<method>-<revision>/
├── METHOD_CARD.md
├── submission.json
└── results/*.json
```

See the [submission guide](../docs/SUBMITTING_RESULTS.md) and
[`submission.example.json`](submission.example.json). Do not commit datasets,
weights, videos, credentials, or machine-specific paths here.

`leaderboard.json` is generated from every validated `submission.json` by:

```bash
python scripts/build_community_leaderboard.py
```

Do not edit leaderboard values by hand. CI checks that the registry and the
community table in the repository README match the submitted result traces.

Passing validation means the submitted evidence is internally consistent. It
does not by itself mean that maintainers reproduced or endorsed the method.
Verification status is reported explicitly as `self-reported`, `reproducible`,
or `maintainer-verified`.
