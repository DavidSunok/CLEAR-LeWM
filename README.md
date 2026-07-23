<p align="center">
  <img src="assets/readme_hero_v03_fast.png" width="100%" alt="CLEAR-LeWM task-semantic evaluation and audited FAST loader">
</p>

<h1 align="center">CLEAR-LeWM</h1>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.5.0-f26b5e" alt="v0.5.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-101828" alt="MIT License"></a>
  <a href="tests"><img src="https://img.shields.io/badge/tests-48%20passed-15803d" alt="48 tests passed"></a>
  <a href="results/v0.5"><img src="https://img.shields.io/badge/protocol-v0.5-e5b94f" alt="v0.5 Moderate protocol"></a>
  <a href="manifests/v0.5"><img src="https://img.shields.io/badge/tasks-4-65ae6e" alt="4 tasks"></a>
</p>

<p align="center">
  <strong>Official task semantics · Corrected physics · No pre-solved pairs · Fixed paired manifests</strong><br>
  CLEAR-LeWM freezes comparable goals, repairs demonstrated evaluator defects, audits random floors,
  and records enough provenance to reproduce every reported success.
</p>

<p align="center">
  <strong>FAST training I/O: 17.87× geometric-mean loader speedup across four tasks</strong><br>
  <sub>5.79× PushT · 37.72× Cube · 30.77× Reacher · 15.15× TwoRoom · exact source-equivalence audits · <a href="PERFORMANCE.md">Performance audit</a></sub>
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
  <a href="https://davidsunok.github.io/CLEAR-LeWM/"><strong>Website</strong></a> ·
  <a href="#results"><strong>Results</strong></a> ·
  <a href="#two-auditable-modes"><strong>Modes</strong></a> ·
  <a href="#01-pusht"><strong>Task Guides</strong></a> ·
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="EVALUATION_SPEC.md"><strong>Specification</strong></a> ·
  <a href="docs/SUBMITTING_RESULTS.md"><strong>Submit Results</strong></a> ·
  <a href="checkpoints/official-v0.3.json"><strong>Checkpoints</strong></a>
</p>

<h2 align="center">Four tasks. Eight judgements. One clear standard.</h2>

<p align="center">
  <a href="assets/showcase/clear_lewm_v03_overview_1080p.mp4">
    <img src="assets/showcase/clear_lewm_v03_overview_preview.gif" width="100%" alt="Historical v0.3 evaluator audit showcase">
  </a>
</p>

<p align="center">
  <strong>Historical v0.3 mechanism audit.</strong><br>
  The current normative protocol is v0.5 Moderate; archived v0.3 media and results remain available for reproducibility.
</p>

> [!IMPORTANT]
> CLEAR-LeWM is an independent community evaluation project, not an official
> LeWM release. The historical `official` track is preserved unchanged;
> `moderate` makes the explicitly versioned v0.5 claim. Archived v0.3 Strict
> artifacts keep their embedded protocol and are not silently reinterpreted.

> [!WARNING]
> Published comparisons use **solver batch size 1**. Batch 16 is a development
> throughput mode: it changes CEM random-number ordering and flipped 99/400
> episode outcomes in a controlled four-task audit.

## What v0.5 fixes

<table>
  <tr>
    <td width="25%"><a href="docs/tasks/PUSHT.md"><strong>PushT · aligned goal state</strong></a><br>Keep the official pusher + T pose contract.<br><sub><a href="docs/tasks/PUSHT.md">Manifest + rollout gates →</a></sub></td>
    <td width="25%"><a href="docs/tasks/CUBE.md"><strong>Cube · official object goal</strong></a><br>Score cube position, not robot pose or added orientation.<br><sub><a href="docs/tasks/CUBE.md">Manifest + rollout gates →</a></sub></td>
    <td width="25%"><a href="docs/tasks/REACHER.md"><strong>Reacher · joint topology</strong></a><br>Wrap the periodic shoulder, not the bounded wrist.<br><sub><a href="docs/tasks/REACHER.md">Manifest + rollout gates →</a></sub></td>
    <td width="25%"><a href="docs/tasks/TWOROOM.md"><strong>TwoRoom · clean physics</strong></a><br>Reject polluted source windows and sweep the full disk.<br><sub><a href="docs/tasks/TWOROOM.md">Manifest + rollout gates →</a></sub></td>
  </tr>
</table>

The four contracts live in the evaluator, fixed manifests, tests, reference
outputs, and task guides. All older artifacts remain executable because every
manifest embeds its complete protocol.

## Results

Official high-epoch LeWM checkpoints are reevaluated on 100 deterministic v0.5
Moderate pairs, seed 42, `300 x 30` CEM, top-k 30, solver batch 1. Model and
random use the exact same manifest. The checked-in v0.5 result files are the
source of truth for the table below.

| Task | Official LeWM | Paired random | Excess |
|---|---:|---:|---:|
| **PushT** | **88%** | 3% | **+85 pp** |
| **Cube** | **51%** | 15% | **+36 pp** |
| **Reacher** | **40%** | 5% | **+35 pp** |
| **TwoRoom** | **81%** | 6% | **+75 pp** |

Historical v0.3 Moderate/Strict outputs remain under
[`results/v0.3/`](results/v0.3/) with their immutable manifests in
[`manifests/v0.3/`](manifests/v0.3/). They answer different task definitions
and must not be mixed into a v0.5 ranking column.

## Community results

CLEAR-LeWM accepts public method results through auditable pull requests. A
submission keeps fixed versioned manifests, records its training-data track and
inference budget, includes every episode outcome, and passes automated hash,
protocol, provenance, and metric checks. Results are labeled `self-reported`,
`reproducible`, or `maintainer-verified`; CI validity is never presented as an
independent reproduction.

[Submit a result or propose a new data track](docs/SUBMITTING_RESULTS.md).

## v0.5 Moderate

| Task | Pair contract | Success contract |
|---|---|---|
| PushT | episode-balanced, initially unsolved | pusher + block `<20 px`, T angle `<20 deg`, first hit |
| Cube | episode-balanced, initially unsolved | cube position `<=4 cm`, first hit |
| Reacher | episode-balanced, initially unsolved | periodic shoulder + bounded wrist, max `<0.05 rad`, first hit |
| TwoRoom | cross-room, endpoint-clear, clean `+25` source window | swept-disk runtime, endpoint `<16 px`, first hit |

No Moderate task uses an arbitrary minimum displacement or multi-step hold.
Exact definitions are normative in [`EVALUATION_SPEC.md`](EVALUATION_SPEC.md).

<p align="center">
  <strong>TASK-BY-TASK EVALUATION</strong><br>
  Each guide separates the released predicate, the failure mode, the corrected rule, and the paired random floor.
</p>

## 01. PushT

**Keep the benchmark target aligned with the full goal image.**

> **FAST input: 5.79x loader throughput vs Lance.** Exact-audited pixels,
> action chunks, proprio, state, and episode boundaries.

v0.5 keeps the released combined pusher-plus-block position error and wrapped
T-block angle. It removes only initially solved pairs and episode imbalance;
it does not redefine PushT as a block-only endpoint task.

<p align="center">
  <img src="assets/task_gifs/pusht.gif" width="900" alt="PushT object-pose evaluation and rollout trace">
</p>

| Audit question | PushT definition |
|---|---|
| **Evaluation target** | Match the released pusher and T-block goal state. |
| **Manifest correction** | Remove starts already within the exact rollout threshold; sample episodes uniformly. |
| **Success** | Combined position `<20 px`, wrapped T angle `<20 deg`, first hit. |
| **Not added** | No block-only reinterpretation, minimum displacement, or hold. |

[Read the PushT evaluation guide](docs/tasks/PUSHT.md).

## 02. Cube

**Follow the OGBench object-position task.**

> **FAST input: 37.72x loader throughput vs HDF5.** Exact-audited pixels,
> action chunks, observations, merged 19-D proprio, and episode boundaries.

v0.5 scores cube center position within 4 cm, exactly as the OGBench task
defines it. Cube orientation and terminal robot/gripper pose remain diagnostics,
not Moderate success conditions.

<p align="center">
  <img src="assets/task_gifs/cube.gif" width="900" alt="Cube symmetry-aware evaluation and rollout trace">
</p>

| Audit question | Cube definition |
|---|---|
| **Evaluation target** | Move the cube center to the target. |
| **Manifest correction** | Remove starts already within 4 cm; sample episodes uniformly. |
| **Success** | Cube position `<=0.04 m`, first hit. |
| **Not scored** | Cube orientation and terminal robot/gripper pose. |

[Read the Cube evaluation guide](docs/tasks/CUBE.md).

## 03. Reacher

**Use the topology of each joint, not one rule for both.**

> **FAST input: 30.77x loader throughput vs HDF5.** Exact-audited pixels,
> action chunks, observations, and episode boundaries.

The shoulder is periodic and uses shortest wrapped error. The wrist is bounded
and uses raw absolute error. Moderate preserves the released `<0.05 rad`
first-hit meaning; holding remains a separately labeled diagnostic.

<p align="center">
  <img src="assets/task_gifs/reacher.gif" width="900" alt="Reacher wrapped-angle first-hit evaluation">
</p>

| Audit question | Reacher definition |
|---|---|
| **Evaluation target** | Reach the goal joint configuration. |
| **Topology correction** | Wrap the unbounded shoulder; do not wrap the limited wrist. |
| **Success** | Maximum corrected joint error `<0.05 rad`, first hit. |
| **Not added** | No relaxed threshold or hold requirement. |

[Read the Reacher evaluation guide](docs/tasks/REACHER.md).

## 04. TwoRoom

**Correct the environment defect without inventing a second task.**

> **FAST input: 15.15x loader throughput vs HDF5.** Exact-audited pixels,
> action chunks, proprio, and episode boundaries.

Moderate restores canonical cross-room sampling, rejects any `+25` source
window containing an illegal transition, and installs continuous swept-disk
collision at runtime. Success itself stays the released endpoint distance
`<16 px`, first hit.

<p align="center">
  <img src="assets/task_gifs/tworoom.gif" width="900" alt="TwoRoom legal swept-circle rollout and route-valid distance trace">
</p>

| Audit question | TwoRoom definition |
|---|---|
| **Evaluation target** | Reach a goal sampled in the other room. |
| **Data correction** | Require clear endpoints and 25/25 legal source transitions. |
| **Physics correction** | Resolve each requested move as a complete swept disk. |
| **Success** | Endpoint distance `<16 px`, first hit; route remains an audit field. |

[Read the TwoRoom guide](docs/tasks/TWOROOM.md) or
[watch the dedicated 1080p topology film](assets/showcase/tworoom_topology_1080p.mp4).

## Quick Start

### 1. Install

```bash
git clone --recurse-submodules https://github.com/DavidSunok/CLEAR-LeWM.git
cd CLEAR-LeWM
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,lewm]'
```

### 2. Prepare verified official checkpoints

```bash
python scripts/prepare_official_checkpoints.py --cache-dir "$STABLEWM_HOME"
```

The downloader pins four upstream revisions, checks the immutable source
weight SHA-256, reconstructs the model, and strictly loads **303/303 tensors**.
The release registry is [`checkpoints/official-v0.3.json`](checkpoints/official-v0.3.json).
Binary weights are intentionally not committed to ordinary Git.

### 3. Evaluate random and model on identical pairs

```bash
clear-lewm evaluate \
  --manifest manifests/v0.5/tworoom/moderate-seed42-n100.json \
  --policy random --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/tworoom.h5 \
  --solver-batch-size 1 \
  --output results/tworoom-v05-moderate-random.json

clear-lewm evaluate \
  --manifest manifests/v0.5/tworoom/moderate-seed42-n100.json \
  --policy official/tworoom/weights.pt --policy-label official-lewm \
  --cache-dir "$STABLEWM_HOME" \
  --dataset-path /path/to/tworoom.h5 \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/tworoom-v05-moderate-random.json \
  --output results/tworoom-v05-moderate-lewm.json
```

The output contains the manifest hash, embedded criterion, per-episode outcomes,
paired random gain, checkpoint revision and hashes, environment fingerprint,
solver settings, and task-specific topology diagnostics.

### Planning mode must be explicit

For representation-only planning comparisons, pass `--actor-warmstart off`.
This records audited **pure-CEM**. The default `auto` preserves checkpoint
configuration and may initialize CEM from an action prior.

For action-head-only evaluation, use `--inference-mode direct` with
`--actor-warmstart on`, an explicit `--direct-target-mode`, and the checkpoint's
verified runtime directory. CLEAR records the solver source SHA-256.

## Reproducibility contract

Every matched comparison must share:

1. manifest and dataset fingerprint;
2. environment, physics, action bounds, and evaluation budget;
3. policy, solver, and environment seeds;
4. planner samples, iterations, top-k, and **batch size 1**;
5. task criterion and protocol version.

MuJoCo, Pymunk, dm-control, Gymnasium, task source, PyTorch, CUDA, cuDNN,
accelerator, evaluator source, and custom Hydra targets are fingerprinted.
See [`docs/RUNTIME_REPRODUCIBILITY.md`](docs/RUNTIME_REPRODUCIBILITY.md).

## Official compatibility and history

`official` preserves upstream row-uniform sampling, initially solved starts,
first-hit predicates, and the released off-by-one sampling range. It is useful
for historical comparison, not a strong completion benchmark.

The v0.2 manifests and 24 reference outputs remain under
[`manifests/`](manifests/) and [`results/reference/`](results/reference/).
Their embedded protocols remain executable. v0.3 artifacts remain under their
versioned directories, and v0.5 never mutates an archived result in place.

## FAST and runtime performance

FAST is an audited training I/O path, not a new dataset. It decodes once,
stores row-major memmaps, preserves complete action chunks, and verifies tensor
equivalence against the source reader.

| Training input | Source samples/s | FAST samples/s | Paired speedup |
|---|---:|---:|---:|
| PushT / Lance | 672.3 | 3812.0 | **5.79x** |
| Cube / HDF5 | 119.5 | 4426.5 | **37.72x** |
| Reacher / HDF5 | 143.0 | 4362.1 | **30.77x** |
| TwoRoom / HDF5 | 279.2 | 4291.4 | **15.15x** |

The four-task aggregate is **17.87x geometric-mean steady-state loader
speedup**; the arithmetic mean is 22.36x. These numbers exclude one-time
conversion and model compute. A separate historical PushT run observed **1.8x
end-to-end training throughput**; development CEM batch 16 observed **1.49x
evaluation throughput** but changes planner trajectories and is never used for
published tables.

Batch 16 is faster but not numerically equivalent. Published reference tables
and model selection stay at batch 1. Full measurements and negative results are
in [`PERFORMANCE.md`](PERFORMANCE.md).

## Data contract

| PushT | Cube | Reacher | TwoRoom |
|---|---|---|---|
| Lance -> exact-audited FAST | HDF5 -> exact-audited FAST | HDF5 -> exact-audited FAST | HDF5 -> exact-audited FAST |

Evaluation manifests always reference canonical HDF5 row IDs. Training may use
HDF5, Lance, or FAST only after episode boundaries, action chunks,
normalization, and RGB tensors pass [`DATA_SPEC.md`](DATA_SPEC.md).

## Repository map

| Path | Purpose |
|---|---|
| [`clear_lewm/`](clear_lewm) | task semantics, topology, manifests, metrics, and runner |
| [`manifests/v0.5/`](manifests/v0.5) | canonical v0.5 Moderate pair sets |
| [`results/v0.5/`](results/v0.5) | paired v0.5 official-LeWM and random outputs |
| [`manifests/v0.3/`](manifests/v0.3) | archived v0.3 Moderate and Strict pair sets |
| [`results/v0.3/`](results/v0.3) | 16 archived v0.3 audited outputs |
| [`docs/tasks/`](docs/tasks) | task-by-task evaluation guides |
| [`checkpoints/`](checkpoints) | official revision and hash registry |
| [`assets/showcase/`](assets/showcase) | 1080p overview and topology films |
| [`scripts/build_v03_media.py`](scripts/build_v03_media.py) | synchronized task GIF and 1080p comparison-film generator |
| [`assets/media_sources/`](assets/media_sources) | recorded comparison traces and result summaries used by the media builder |
| [`scripts/`](scripts) | checkpoint, FAST, environment, and remaining utility scripts |
| [`third_party/le-wm/`](third_party/le-wm) | pinned upstream LeWM submodule |

## Scope and attribution

LeWorldModel remains authored and licensed by its upstream authors. PushT,
Cube, and Reacher media sample RGB and metrics from the same canonical episode
indices; TwoRoom uses recorded rollout trajectories. See
[`NOTICE.md`](NOTICE.md) and [`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md).
