# CLEAR-LeWM

**C**ontrolled, **L**eakage-aware, **E**pisode-balanced, **A**uditable and
**R**eproducible evaluation for LeWM-compatible latent world models.

CLEAR-LeWM preserves the original LeWorldModel evaluation protocol for
historical comparison and adds deterministic, difficulty-controlled tracks for
stronger scientific claims. The project separates protocol compatibility from
protocol validity: old numbers remain reproducible, while new numbers expose
trivial goals, random-policy floors, data overlap, and task-specific success
criteria.

> CLEAR-LeWM is an independent community project and is not an official LeWM
> release.

## Why this exists

The official LeWM benchmark samples a state from an offline trajectory, uses
the state 25 steps later as the goal, and allows a 50-step evaluation budget.
That is a useful reachable-goal protocol, but several details complicate the
interpretation of raw success rate:

- start-goal rows are sampled rather than episodes, so long episodes receive
  more weight;
- the configured evaluation data are also used for training unless users make
  a separate split;
- initially solved start-goal pairs are retained;
- the random action policy is not seeded by the official evaluation script;
- OGBench-Cube success checks only a 4 cm position threshold, despite receiving
  a target quaternion, and does not require a grasp or stable placement;
- success is a first-hit event rather than a final or sustained condition.

The LeWM paper itself reports 48% random-policy success on Cube. On the public
Cube dataset snapshot used in our audit, 38.38% of all valid `(t, t+25)` pairs
already satisfy the 4 cm position criterion before control begins. See
[`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md) for the reproducible audit.

On the checked-in `clear-id` manifests, deterministic random baselines are 3%
on PushT, 9% on Reacher, 22% on TwoRoom, and **1% on Cube**. The strict Cube
protocol removes the approximately 49% random floor observed under the original
position-only sampling and criterion.

## Protocols

| Track | Split | Sampling | Initial successes | Cube criterion |
|---|---|---|---|---|
| `official-compat` | all data | row-uniform | retained | position <= 4 cm, first hit |
| `clear-id` | all data | episode-balanced | removed | position <= 4 cm, orientation <= 15 deg, held 5 steps |
| `clear-standard` | deterministic 20% held-out episodes | episode-balanced | removed | position <= 4 cm, orientation <= 15 deg, held 5 steps |
| `clear-hard` | same held-out split | episode-balanced and distance-filtered | removed | strict pose criterion, held 5 steps |

The complete normative definition is in
[`EVALUATION_SPEC.md`](EVALUATION_SPEC.md).

## Installation

```bash
git clone --recurse-submodules https://github.com/DavidSunok/CLEAR-LeWM.git
cd CLEAR-LeWM
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Install the optional LeWM runtime only when evaluating a checkpoint:

```bash
pip install -e '.[lewm]'
```

The official LeWM source is pinned as a Git submodule. If the repository was
cloned without submodules, run:

```bash
git submodule update --init --recursive
```

## Dataset audit

Auditing reads only state columns; image tensors are never loaded:

```bash
clear-lewm audit /path/to/cube_single_expert.h5 \
  --task cube \
  --output results/cube_audit.json
```

The report includes the initial-success floor and start-goal difficulty
quantiles.

## Fixed manifests

Generate a reusable set of 100 held-out, episode-balanced pairs:

```bash
clear-lewm manifest /path/to/cube_single_expert.h5 \
  --task cube \
  --protocol clear-standard \
  --num-eval 100 \
  --seed 42 \
  --output manifests/cube/clear-standard-seed42-n100.json
```

Every method and baseline must use the same checked-in manifest. Manifests
contain episode IDs, start/goal steps, row indices, difficulty, protocol
parameters, and a dataset metadata fingerprint. Use `--full-sha256` when
publishing a final dataset artifact; hashing the 40-100 GB image datasets is
intentionally opt-in.

Released LeWM checkpoints were trained on the full public datasets. Evaluate
them with `official-compat` or `clear-id`; they are not eligible for a held-out
claim under `clear-standard` unless retrained with the manifest-defined held-out
episodes removed.

## Evaluation

First run the deterministic random baseline:

```bash
clear-lewm evaluate \
  --manifest manifests/cube/clear-standard-seed42-n100.json \
  --policy random \
  --cache-dir "$STABLEWM_HOME" \
  --output results/cube_random.json
```

Then evaluate a LeWM-compatible checkpoint on the identical pairs:

```bash
clear-lewm evaluate \
  --manifest manifests/cube/clear-standard-seed42-n100.json \
  --policy cube/lewm \
  --cache-dir "$STABLEWM_HOME" \
  --random-results results/cube_random.json \
  --output results/cube_lewm.json
```

For Cube, `clear-standard` and `clear-hard` patch only the success predicate:
the model, observations, action space, solver, and dynamics remain unchanged.

## Reported metrics

CLEAR-LeWM reports raw SR with a bootstrap confidence interval, excess over
random in percentage points, paired gain confidence intervals, and
random-normalized success:

```text
normalized_SR = (SR_method - SR_random) / (100 - SR_random) * 100
```

Random normalization does not replace strict success criteria. It makes task
floors visible while the strict tracks test whether a policy solved a
non-trivial goal.

## Upstream attribution

LeWorldModel is included only as a pinned public Git submodule and retains its
MIT license and authorship. CLEAR-LeWM contains no private model code,
checkpoints, datasets, or unpublished experiment logs. See
[`NOTICE.md`](NOTICE.md).

## Contributing

Protocol changes require a versioned specification update, synthetic regression
tests, and a migration note. Please see [`CONTRIBUTING.md`](CONTRIBUTING.md).
