<p align="center">
  <img src="assets/readme_hero_v08_fast.png" width="100%" alt="CLEAR-LeWM v0.8 Moderate and Strict task-semantic evaluation">
</p>

<p align="center"><sub>v0.8 Strict SR is shown as official LeWM / paired random, rounded from the mean over seeds 0, 1, and 42.</sub></p>

<h1 align="center">CLEAR-LeWM</h1>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.8.0-f26b5e" alt="v0.8.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-101828" alt="MIT License"></a>
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-pytest%20%2B%20ruff-15803d" alt="pytest and ruff CI"></a>
  <a href="results/v0.8"><img src="https://img.shields.io/badge/protocol-v0.8-e5b94f" alt="v0.8 protocol"></a>
  <a href="manifests/v0.8"><img src="https://img.shields.io/badge/modes-Moderate%20%7C%20Strict-65ae6e" alt="Moderate and Strict"></a>
</p>

<p align="center">
  <strong>Minimal evaluator repair when comparability matters. Task-semantic precision when completion matters.</strong><br>
  CLEAR-LeWM removes pre-solved cases, repairs demonstrated evaluator defects,
  freezes paired manifests, and exposes every success decision for audit.
</p>

<p align="center">
  <a href="mailto:luoliibaqi4747@gmail.com"><strong>Junhan Sun</strong></a><sup>1</sup>
  &nbsp;&nbsp;
  <strong>Guofeng Zhang</strong><sup>1,&#8224;</sup>
  &nbsp;&nbsp;
  <strong>Hao Zhao</strong><sup>2,&#8224;</sup><br>
  <sub><sup>1</sup>State Key Laboratory of CAD&amp;CG, Zhejiang University &nbsp;|&nbsp;
  <sup>2</sup>Tsinghua University &nbsp;|&nbsp;
  <sup>&#8224;</sup>Corresponding authors</sub>
</p>

<p align="center">
  <a href="https://davidsunok.github.io/CLEAR-LeWM/"><strong>Website</strong></a> &middot;
  <a href="https://github.com/DavidSunok/CLEAR-LeWM/releases"><strong>Releases</strong></a> &middot;
  <a href="#reference-results"><strong>Results</strong></a> &middot;
  <a href="#two-auditable-modes"><strong>Modes</strong></a> &middot;
  <a href="EVALUATION_SPEC_V08.md"><strong>Specification</strong></a> &middot;
  <a href="docs/SUBMITTING_RESULTS.md"><strong>Submit Results</strong></a> &middot;
  <a href="checkpoints/official-v0.5.json"><strong>Checkpoints</strong></a>
</p>

<p align="center">
  <a href="assets/showcase/clear_lewm_v08_overview_1080p.mp4">
    <img src="assets/showcase/clear_lewm_v08_overview_preview.gif" width="100%" alt="CLEAR-LeWM v0.8 Moderate and Strict overview">
  </a>
</p>

> [!IMPORTANT]
> CLEAR-LeWM is an independent community evaluation project, not an official
> LeWM release. It reevaluates pinned official LeWM checkpoints and preserves
> their provenance. Official LeWM and paired random define the reference table;
> related public checkpoints are reported separately as independently rerun
> comparisons.

> [!NOTE]
> v0.8 closes a Reacher evaluator leak: dm-control task termination is disabled
> before rollout and reapplied after reset or model recompilation, so CLEAR owns
> the full success decision. Historical v0.5 remains available as the immutable
> [`v0.5.1` release](https://github.com/DavidSunok/CLEAR-LeWM/releases/tag/v0.5.1)
> and under [`results/v0.5/`](results/v0.5/).

> [!WARNING]
> Published comparisons use **solver batch size 1**. Batch 16 changes CEM
> random-number ordering and is a development throughput mode, not a matched
> reference setting.

<p align="center">
  <a href="https://davidsunok.github.io/CLEAR-LeWM/#results">
    <img src="assets/community_model_comparison_v08.png" width="100%" alt="Matched v0.8 Moderate and Strict success rates for Official LeWM, DINOv2 No-Proprio LeWM, and GCBC Joint LeWM">
  </a>
</p>

<p align="center"><sub>RTX 4090, seeds 0/1/42, 100 episodes each, pure CEM 300 x 30. Shared released tasks only.</sub></p>

## Why CLEAR-LeWM

The historical stack mixes genuinely difficult control with evaluator effects:
initially solved start-goal pairs, incorrect Reacher angle topology, early
dm-control termination before CLEAR can score the rollout, and a TwoRoom
rewrite whose endpoint-only collision check can admit an invalid wall crossing.
Cube also has a high random floor because many sampled windows do not move the
cube.

CLEAR-LeWM separates two scientific questions instead of forcing one rule to
answer both.

## Two auditable modes

| Mode | Scientific question | Design rule |
|---|---|---|
| **Moderate** | Does a method improve LeWM after fixing evaluator bugs and trivial cases? | Change as little as possible; preserve released PushT and Cube predicates. |
| **Strict** | Does the rollout precisely complete the task-relevant physical goal? | Score the object or endpoint, tighten geometry, and require short persistence where appropriate. |

| Task | v0.8 Moderate | v0.8 Strict |
|---|---|---|
| **PushT** | pusher + T position `<20 px`; T angle `<20 deg`; first hit | T only `<10 px / 10 deg`; hold 3 |
| **Cube** | cube center `<=4 cm`; first hit | cube center `<=3 cm` + 24-fold orientation `<=15 deg`; hold 3 |
| **Reacher** | periodic shoulder + bounded wrist `<0.05 rad`; first hit | fingertip endpoint `<=1 cm`; hold 2 |
| **TwoRoom** | clean cross-room pair; swept disk; endpoint `<16 px` | valid legal crossing + goal side + endpoint `<8 px` |

Moderate is the closest corrected continuation of the released benchmark.
Strict is the stronger semantic claim. They must be reported as separate
columns. Exact inequalities and runtime gates are normative in
[`EVALUATION_SPEC_V08.md`](EVALUATION_SPEC_V08.md).

## Reference results

The primary table uses pinned official high-epoch LeWM checkpoints and paired
random policies. All runs were independently executed on RTX 4090 GPUs with
100 episodes, `300 x 30` CEM, top-k 30, solver batch size 1, and strict 303/303
tensor loading.

### Moderate: seeds 0, 1, and 42

Mean +/- sample standard deviation across complete paired runs:

| Task | Official LeWM | Paired random | Excess |
|---|---:|---:|---:|
| **PushT** | **86.67 +/- 0.58%** | 4.00 +/- 1.00% | **+82.67 pp** |
| **Cube** | **50.33 +/- 3.79%** | 15.67 +/- 6.03% | **+34.67 pp** |
| **Reacher** | **79.67 +/- 0.58%** | 7.33 +/- 2.08% | **+72.33 pp** |
| **TwoRoom** | **83.00 +/- 5.29%** | 6.67 +/- 1.15% | **+76.33 pp** |

### Strict: seeds 0, 1, and 42

Mean +/- sample standard deviation across complete paired runs:

| Task | Official LeWM | Paired random | Mean excess |
|---|---:|---:|---:|
| **PushT** | **70.67 +/- 5.51%** | 5.00 +/- 1.73% | **+65.67 pp** |
| **Cube** | **21.67 +/- 4.73%** | 6.00 +/- 2.65% | **+15.67 pp** |
| **Reacher** | **87.00 +/- 1.00%** | 8.00 +/- 6.24% | **+79.00 pp** |
| **TwoRoom** | **51.33 +/- 3.21%** | 1.67 +/- 2.89% | **+49.67 pp** |

The JSON files in [`results/v0.8/runs/`](results/v0.8/runs/) are the source of truth.
They include all episode outcomes, manifest hashes, criteria, solver settings,
environment fingerprints, checkpoint hashes, and topology diagnostics.

## Task guides

### 01. PushT

<p align="center"><img src="assets/task_gifs/pusht_v08.gif" width="900" alt="PushT v0.8 object-pose trace"></p>

Moderate preserves the complete pusher-plus-block goal state. Strict asks the
task-semantic question: is the T itself placed correctly? A full-image latent
cost can therefore differ from Strict completion, and methods should disclose
their planning target. [Read the PushT guide](docs/tasks/PUSHT.md).

### 02. Cube

<p align="center"><img src="assets/task_gifs/cube_v08.gif" width="900" alt="Cube v0.8 position and symmetry-aware pose trace"></p>

Moderate follows OGBench's 4 cm cube-position task. Strict adds a 3 cm position
gate and 15 degree orientation modulo all 24 proper cube rotations. Neither
mode scores terminal robot pose. [Read the Cube guide](docs/tasks/CUBE.md).

### 03. Reacher

<p align="center"><img src="assets/task_gifs/reacher_v08.gif" width="900" alt="Reacher v0.8 joint-topology and endpoint trace"></p>

Moderate repairs the shoulder/wrist topology while preserving joint matching.
Strict scores the physical fingertip endpoint for two consecutive steps.
Both modes disable the environment's own task termination throughout rollout,
including after reset and recompilation, so only the declared CLEAR predicate
can terminate scoring.
[Read the Reacher guide](docs/tasks/REACHER.md).

### 04. TwoRoom

<p align="center"><img src="assets/task_gifs/tworoom_v08.gif" width="900" alt="TwoRoom v0.8 swept-disk route trace"></p>

Both modes reject polluted source windows and execute corrected swept-disk
physics. Strict additionally requires a legal room crossing, goal-side arrival,
and an 8 px endpoint. [Read the TwoRoom guide](docs/tasks/TWOROOM.md).

## Quick start

```bash
git clone --recurse-submodules https://github.com/DavidSunok/CLEAR-LeWM.git
cd CLEAR-LeWM
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,lewm]'
python scripts/prepare_official_checkpoints.py --cache-dir "$STABLEWM_HOME"
```

Evaluate the identical Strict pair set with random and official LeWM:

```bash
clear-lewm evaluate \
  --manifest manifests/v0.8/tworoom/strict-seed42-n100.json \
  --policy random --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/tworoom.h5 \
  --solver-batch-size 1 \
  --output results/tworoom-v08-strict-random.json

clear-lewm evaluate \
  --manifest manifests/v0.8/tworoom/strict-seed42-n100.json \
  --policy official/tworoom/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/tworoom.h5 \
  --num-samples 300 --n-steps 30 --topk 30 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/tworoom-v08-strict-random.json \
  --output results/tworoom-v08-strict-lewm.json
```

Canonical v0.8 reference runs use pure CEM with `--actor-warmstart off`; no
alternative inference contract is mixed into the reference table.

## Audited FAST training I/O

FAST is an optional derived reader, not a new dataset. It decodes once into
verified row-major memory maps while preserving complete action chunks and
episode boundaries.

| Training input | Source samples/s | FAST samples/s | Paired speedup |
|---|---:|---:|---:|
| PushT / Lance | 672.3 | 3812.0 | **5.79x** |
| Cube / HDF5 | 119.5 | 4426.5 | **37.72x** |
| Reacher / HDF5 | 143.0 | 4362.1 | **30.77x** |
| TwoRoom / HDF5 | 279.2 | 4291.4 | **15.15x** |

The four-task geometric-mean loader speedup is **17.87x**. These figures
exclude conversion and model compute. See [`PERFORMANCE.md`](PERFORMANCE.md).

## Reproducibility contract

Every matched comparison must share the manifest, dataset fingerprint,
environment, policy seed, control budget, solver budget, solver batch size,
and protocol mode. MuJoCo, Pymunk, dm-control, Gymnasium, PyTorch, CUDA, cuDNN,
task source, checkpoint source, and evaluator source are fingerprinted.

The pinned official source revisions and hashes are in
[`checkpoints/official-v0.5.json`](checkpoints/official-v0.5.json). Binary
weights and datasets remain outside ordinary Git.

## Community results

Public methods may submit auditable Moderate and/or Strict result bundles.
CI verifies structure, canonical manifest hashes, trace arithmetic, provenance,
and topology. It does not imply independent reproduction or endorsement.

### Independently rerun v0.8 checkpoints

The following related public checkpoints were rerun on the same RTX 4090
stack as official LeWM. Values are mean +/- sample standard deviation over
seeds 0, 1, and 42, with 100 episodes per seed. Neither release includes a
Reacher checkpoint.

| Method | Task | Moderate | Strict |
|---|---|---:|---:|
| **DINOv2 No-Proprio LeWM** | PushT | 7.00 +/- 3.61% | 8.67 +/- 1.53% |
|  | Cube | 44.67 +/- 2.52% | 13.00 +/- 3.00% |
|  | TwoRoom | 43.67 +/- 8.96% | 25.67 +/- 2.08% |
| **GCBC Joint LeWM** | PushT | 5.33 +/- 3.51% | 7.33 +/- 1.53% |
|  | Cube | 17.67 +/- 6.43% | 3.67 +/- 2.08% |
|  | TwoRoom | 16.00 +/- 3.00% | 6.67 +/- 1.15% |

Full traces and hashes are included under [`results/v0.8/runs/`](results/v0.8/runs/).

### Historical v0.5 submissions

<!-- community-leaderboard:start -->

Canonical community entries use policy seed 42 and 100 episodes per
task/mode. Values are **model / paired random (excess)**. CI validates
the bundle structure, canonical manifests, trace arithmetic, provenance,
and topology; the verification label states whether execution was
independently reproduced.

| Method | Task | Moderate | Strict | Verification |
|---|---|---:|---:|---|
| [DINOv2 No-Proprio LeWM](submissions/zerotul782231/dinov2-no-proprio-lewm-141dc536/METHOD_CARD.md) | PushT | 8% / 3% (+5 pp) | 7% / 7% (0 pp) | self-reported; [@zerotul782231](https://github.com/zerotul782231) |
|  | Cube | 43% / 15% (+28 pp) | 17% / 8% (+9 pp) |  |
|  | Reacher | - | - |  |
|  | TwoRoom | 55% / 6% (+49 pp) | 26% / 0% (+26 pp) |  |
| [GCBC Joint LeWM](submissions/zerotul782231/gcbc-joint-lewm-141dc536/METHOD_CARD.md) | PushT | 9% / 3% (+6 pp) | 9% / 7% (+2 pp) | self-reported; [@zerotul782231](https://github.com/zerotul782231) |
|  | Cube | 16% / 15% (+1 pp) | 3% / 8% (-5 pp) |  |
|  | Reacher | - | - |  |
|  | TwoRoom | 15% / 6% (+9 pp) | 9% / 0% (+9 pp) |  |

A dash means that task/mode was not submitted. Supplementary evidence
that does not match the canonical manifest/seed contract remains in each
method card and is not mixed into this table.

<!-- community-leaderboard:end -->

[Read the submission guide](docs/SUBMITTING_RESULTS.md).

## Repository map

| Path | Purpose |
|---|---|
| [`clear_lewm/`](clear_lewm) | evaluator, manifests, task metrics, topology, submissions |
| [`manifests/v0.8/`](manifests/v0.8) | canonical Moderate/Strict manifests |
| [`results/v0.8/`](results/v0.8) | audited RTX 4090 reference and related-checkpoint results |
| [`results/v0.5/`](results/v0.5) | immutable historical H200 archive |
| [`submissions/leaderboard.json`](submissions/leaderboard.json) | generated community-result registry |
| [`docs/tasks/`](docs/tasks) | task objectives, gates, and reproduction commands |
| [`scripts/build_v08_media.py`](scripts/build_v08_media.py) | synchronized GIF and 1080p overview generator |
| [`tests/`](tests) | protocol, manifest, runtime, result, and submission regressions |

## License and attribution

CLEAR-LeWM is MIT licensed. Upstream LeWM, stable-worldmodel, OGBench, DMC,
Pymunk, MuJoCo, PLDM, and DINO-WM components remain under their own licenses
and attribution requirements.
