<div align="center">

# CLEAR-LeWM

**Controlled, Leakage-aware, Episode-balanced, Auditable, Reproducible**
evaluation for LeWM-compatible visual world models.

[![Release](https://img.shields.io/github/v/release/DavidSunok/CLEAR-LeWM?color=15803d)](https://github.com/DavidSunok/CLEAR-LeWM/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-101828)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-17%20passed-15803d)](tests)

<img src="assets/protocols.gif" width="848" alt="CLEAR-LeWM protocols across PushT, Cube, Reacher, and TwoRoom">

**One fixed goal manifest. Three explicit levels of success rigor. No silent
benchmark changes.**

</div>

> [!IMPORTANT]
> CLEAR-LeWM is an independent community project, not an official LeWM release.
> The `official` tier exists to reproduce upstream behavior; `moderate` and
> `strict` are corrected evaluation standards.

## Why CLEAR-LeWM

LeWM evaluates a goal sampled 25 steps later from an offline episode with a
50-step control budget. That makes goals reachable, but raw success rate can be
inflated by protocol details:

- sampling rows overweights longer episodes;
- training and evaluation use the same public trajectories by default;
- already-solved start-goal pairs are retained;
- first contact with a loose goal region ends the episode;
- Cube ignores target orientation under the upstream 4 cm position test.

On our canonical snapshots, the control-free initial-success floor is **38.38%
for Cube** and **8.82% for TwoRoom** under upstream predicates. The LeWM paper
also reports a 48% Cube random baseline. CLEAR-LeWM keeps that protocol
reproducible while making stronger claims measurable.

## Three Tiers

| Tier | Pair sampling | Initial success | Temporal rule | Intended claim |
|---|---|---|---|---|
| `official` | upstream row-uniform | retained | first hit | historical compatibility |
| `moderate` | episode-balanced | removed | calibrated target held 2-3 steps | robust in-distribution planning |
| `strict` | balanced + difficulty floor | removed | tighter target held 2-5 steps | conservative task completion |

Task thresholds are shown in the animation and defined normatively in
[`EVALUATION_SPEC.md`](EVALUATION_SPEC.md). Data split is separate: use
`--split all` for released full-data checkpoints, and `--split heldout` only for
models retrained without those held-out episode IDs.

### Calibrated reference

Success rate below is **official LeWM / deterministic random**, using one fixed
100-episode seed-42 manifest per cell and upstream `300 x 30` CEM:

| Task | Official | Moderate | Strict |
|---|---:|---:|---:|
| PushT | **89% / 7%** | **74% / 0%** | **42% / 0%** |
| Reacher | **87% / 16%** | **63% / 6%** | **22% / 0%** |
| TwoRoom | **85% / 30%** | **70% / 17%** | **41% / 1%** |
| Cube | **62% / 47%** | **36% / 3%** | **17% / 2%** |

The stricter tiers reduce absolute SR, as intended, while keeping every good
checkpoint above random. Moderate Cube is especially informative: its raw SR
is lower than Official, but excess over random increases from 15pp to 33pp.
See [`docs/PROTOCOL_CALIBRATION.md`](docs/PROTOCOL_CALIBRATION.md).

## Data Contract

The current four-task shared-encoder training setup uses:

| PushT | Cube | Reacher | TwoRoom |
|---|---|---|---|
| Lance | HDF5 | HDF5 | HDF5 |

These are four fixed **logical** datasets, not four unconstrained files.
Evaluation always uses canonical HDF5 row IDs. HDF5, Lance, or FAST memmap may
be used for training only after episode boundaries, action chunks, transforms,
normalization, and sampled RGB tensors pass the format-equivalence audit.

The later corrected FAST loader is included as
[`clear_lewm/fast_dataset.py`](clear_lewm/fast_dataset.py); the early loader that
skipped transforms is intentionally not reproduced. See
[`DATA_SPEC.md`](DATA_SPEC.md) for canonical names, row counts, known
HDF5/Lance differences, and audit commands.

## FAST and Runtime

FAST is an audited **training I/O optimization**, not a different dataset. It
decodes the authoritative source once, stores each column as a row-major
memmap, keeps every action in the action chunk, applies observation frameskip
only where the source reader does, and still runs the configured transform.
The converter now adds:

- a schema and atomic `complete` flag;
- exact shape, dtype, file-size, episode-length, and offset checks;
- source-safe resume checks that reject mixed snapshots;
- batched source reads and vectorized memmap writes during conversion;
- seeded tensor equality auditing against the source reader.

In a historical PushT H200 run with batch size 128 and 16 loader workers, the
corrected FAST path sustained about **11.0 steps/s**, versus **6.1 steps/s** for
Lance, an observed **1.8x end-to-end training throughput** gain. These were
separate runs and are not a controlled hardware guarantee. In the isolated
default benchmark on the H20-3 node, FAST reached **4,661 samples/s** versus
**504 samples/s** for Lance, a **9.25x loader-only gain** over three fixed-seed
rounds without model compute or image transforms. FAST trades storage for
decode time: the uncompressed PushT snapshot is about 328 GiB. Measure the
actual gain on each storage system with:

```bash
python scripts/preprocess_fast_dataset.py \
  --source pusht_expert_train.lance \
  --cache-dir "$STABLEWM_HOME" \
  --output /local-nvme/pusht-fast

python scripts/audit_fast_dataset.py \
  --source pusht_expert_train.lance \
  --cache-dir "$STABLEWM_HOME" \
  --fast-dir /local-nvme/pusht-fast

python scripts/benchmark_fast_dataset.py \
  --source pusht_expert_train.lance \
  --cache-dir "$STABLEWM_HOME" \
  --fast-dir /local-nvme/pusht-fast
```

Evaluation has a separate bottleneck. Upstream CEM uses one environment per GPU
batch. CLEAR-LeWM supports an opt-in throughput mode and fixes the missing
sample-axis broadcast in canonical LeWM:

```bash
clear-lewm evaluate ... --solver-batch-size 16
```

On H20-3e, PushT Strict with 100 episodes and `300 x 30` CEM reduced the two
CEM calls from **112.8 s** at batch 1 to **75.6 s** at batch 16, a **1.49x**
solver speedup. Larger batches saturated near 1.5x. This mode changes how CEM
random samples are assigned to environments and changed SR in our sweep, so it
must not replace batch-1 reference results. The batch size, matmul mode, CPU
threads, patch status, and measured runtime are recorded in every new result.

The default `CLEAR_LEWM_CPU_THREADS=1` also avoids severe 100-environment CPU
oversubscription. Run one task/protocol/seed per free GPU, keep video disabled
for benchmark sweeps, and see [`PERFORMANCE.md`](PERFORMANCE.md) for the full
audit and safe/unsafe optimization boundary.

## Install

```bash
git clone --recurse-submodules https://github.com/DavidSunok/CLEAR-LeWM.git
cd CLEAR-LeWM
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Install the LeWM runtime only for checkpoint evaluation:

```bash
pip install -e '.[lewm]'
```

## Quick Start

Generate a fixed 100-pair manifest:

```bash
clear-lewm manifest /path/to/tworoom.h5 \
  --task tworoom \
  --protocol moderate \
  --num-eval 100 \
  --seed 42 \
  --output manifests/tworoom/moderate-seed42-n100.json
```

Run the deterministic random floor, then a model on the identical pairs:

```bash
clear-lewm evaluate \
  --manifest manifests/tworoom/moderate-seed42-n100.json \
  --policy random \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/tworoom.h5 \
  --output results/tworoom-random.json

clear-lewm evaluate \
  --manifest manifests/tworoom/moderate-seed42-n100.json \
  --policy official/tworoom \
  --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --random-results results/tworoom-random.json \
  --output results/tworoom-lewm.json
```

Every result stores the manifest hash, exact embedded protocol, criterion,
solver budget, package versions, confidence interval, and per-episode outcome.

## Official Checkpoints

The four official Hugging Face mirrors are revision-pinned and SHA-256 checked.
Prepare the object checkpoints expected by the LeWM runner with:

```bash
python scripts/prepare_official_checkpoints.py --cache-dir "$STABLEWM_HOME"
```

The evaluator policy names are `official/pusht`, `official/cube`,
`official/reacher`, and `official/tworoom`. The script also emits legacy object
checkpoints and supports both ViT state-dict key layouts. Checkpoints are not
added to Git; only hashes and reproducible conversion code belong here.

## Repository Map

| Path | Purpose |
|---|---|
| `clear_lewm/` | manifests, audits, criteria patches, metrics, runner |
| `manifests/` | checked-in 100-pair task/tier manifests |
| `results/reference/` | deterministic random reference results |
| `benchmarks/` | FAST and CEM performance measurements |
| `scripts/` | checkpoint prep, FAST conversion/audit, README asset build |
| `third_party/le-wm/` | pinned upstream LeWM Git submodule |

## Scope and Attribution

LeWorldModel remains authored and licensed by its upstream authors. CLEAR-LeWM
contains no private SICJEPA code, private checkpoints, datasets, or unpublished
experiment logs. The protocol animation uses RGB frames from the public LeWM
datasets solely to identify the benchmark tasks. See [`NOTICE.md`](NOTICE.md)
and [`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md).
