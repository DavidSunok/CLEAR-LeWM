# Runtime reproducibility

## Why the simulator version matters

MuJoCo, dm-control, Gymnasium, OGBench, Pymunk, and the concrete task environment
source participate in environment creation, physics stepping, termination, and
rendering. A version change can alter contact resolution or the exact step on
which a success predicate becomes true. Results from different per-task physics
fingerprints must therefore not be pooled as one matched comparison without a
trajectory-equivalence audit. PushT uses Pymunk rather than MuJoCo; Cube and
Reacher use MuJoCo.

PyTorch, CUDA, cuDNN, and GPU architecture form a separate numerical boundary.
Small floating-point differences can change CEM candidate ranking and amplify
into a different closed-loop trajectory. Physics equality is necessary, but a
paper-quality matched comparison should also use one numerical fingerprint.

## Reproduction classes

CLEAR distinguishes three cases rather than applying one success-rate tolerance
to all reruns:

| Class | Required identity | Interpretation |
|---|---|---|
| Exact reproduction | protocol, data, checkpoint, solver, physics, execution, and numerical fingerprints | all episode outcomes should match; drift is an error to investigate |
| Cross-numerics sensitivity | all fixed identity fields match except the numerical fingerprint | valid portability evidence, but not an exact reproduction claim |
| Incompatible | protocol, data, checkpoint, solver, physics, or execution differs | do not attribute the delta to numerical hardware |

There is no universal `+/- N pp` acceptance threshold across accelerators.
Standard CEM is a discontinuous optimizer: a small cost perturbation can change
an elite boundary, every later proposal, and the simulated trajectory. The
correct report is the three-seed aggregate plus paired episode transitions, not
an architecture-independent bitwise promise.

Compare a candidate result with the checked-in reference trace using:

```bash
python scripts/compare_reproduction_results.py \
  results/v0.5/tworoom-strict-official-lewm-seed0-n100.json \
  /path/to/local/tworoom-strict-official-lewm-seed0-n100.json \
  --output /tmp/tworoom-seed0-comparison.json
```

The report includes identity checks, both-success/both-fail counts, directional
episode flips, a paired bootstrap interval, and an exact McNemar p-value.

## Cross-accelerator diagnostic

Issue #3 supplied a complete v0.5 Strict rerun on an RTX 4090 D. Its fixed
inputs and execution sources match the H200 reference, while its numerical
fingerprint differs. We independently isolated the boundary under Python
3.10.20 and PyTorch 2.6.0+cu124:

- H200 and RTX 4090 generated identical raw CUDA `torch.randn` streams for all
  30 CEM rounds at policy seeds 0, 1, and 42;
- two RTX 4090 devices repeated the same representative FP32 GEMM bit for bit;
- the same GEMM differed between RTX 4090 and H200, even with float32 matmul
  precision set to `highest`.

This rules out CEM RNG drift in that diagnostic and demonstrates the mechanism
that can perturb elite ranking across Ada and Hopper. Reproduce the diagnostic
on another accelerator with:

```bash
python scripts/audit_cross_accelerator_numerics.py --seed 0
```

Enabling deterministic PyTorch algorithms can constrain repeated execution on a
supported stack. It does not promise identical floating-point results across
GPU architectures.

Every new CLEAR result records physics, numerical, and execution fingerprints,
all relevant package versions, the MuJoCo runtime version, CUDA/cuDNN builds,
the NVIDIA driver, accelerator properties, PyTorch determinism/TF32/SDP
controls, relevant CUDA environment variables, and the source hashes of the
world loop, policy, CEM, checkpoint loader, and CLEAR evaluation path. Audit a
result directory with:

```bash
python scripts/audit_result_environments.py results/run \
  --strict-physics --strict-numerics --strict-execution \
  --output results/run/environment-audit.json
```

The CUDA 12.4 stack used for canonical reference runs is pinned in
`requirements/reference-cu124.txt`. Environment directory names are not evidence
of equivalence; compare the recorded fingerprints.

## Evaluator source integrity

The upstream LeWM install command currently requests
`stable-worldmodel[train,env]` without a lockfile, while the
`stable-worldmodel` package leaves PyTorch and torchvision unpinned. Two installs
can therefore resolve to different numerical stacks.

A matching package version is still not enough: a file edited inside
`site-packages` retains the same version string. CLEAR hashes the runtime files
that own `World.evaluate`, `WorldModelPolicy`, CEM, and checkpoint loading. For
the pinned `stable-worldmodel==0.1.0` reference environment, those hashes must
match the published PyPI wheel before any rollout begins. A mismatch fails fast.

`--allow-modified-stable-worldmodel` is an explicit escape hatch for development
experiments. Such a run records the modified source hashes and must not be mixed
into a reference table.

## Custom checkpoint runtimes

Legacy LeWM checkpoints may contain unqualified Hydra targets such as
`jepa.JEPA` and `module.InverseTransitionActor`. Always provide the directory
that owns those files explicitly:

```bash
clear-lewm evaluate \
  --runtime-dir /path/to/model-runtime \
  --upstream-dir third_party/le-wm \
  --strict-checkpoint ...
```

CLEAR places the custom runtime first and upstream LeWM last, resolves every
Hydra target before model construction, and rejects any legacy target whose
source falls outside the requested runtime. The result stores the checkpoint
configuration hash and a source hash for every resolved target.
