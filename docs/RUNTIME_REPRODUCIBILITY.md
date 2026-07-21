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

Every new CLEAR result records both fingerprints, all relevant package versions,
the MuJoCo runtime version, CUDA/cuDNN builds, and accelerator properties. Audit
a result directory with:

```bash
python scripts/audit_result_environments.py results/run \
  --strict-physics --strict-numerics \
  --output results/run/environment-audit.json
```

The CUDA 12.4 stack used for canonical reference runs is pinned in
`requirements/reference-cu124.txt`. Environment directory names are not evidence
of equivalence; compare the recorded fingerprints.

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
