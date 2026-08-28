# Performance and reproducibility

Audit date: 2026-07-23.

Performance changes are divided into two classes. Reference-preserving changes
may be used for published CLEAR results. Throughput modes are useful for model
development but must be reported separately because they can alter numerical
or sampling behavior.

## Reference-preserving defaults

### CPU thread control

`clear-lewm evaluate` defaults PyTorch to one CPU thread. A `World` already
contains one Python environment per evaluation pair; allowing every operation
to open a large BLAS/OpenMP pool causes extreme oversubscription. In the local
100-environment TwoRoom random check, an unrestricted run had not completed
after 27 minutes, while the one-thread run completed in under one minute.

Use `--cpu-threads N` only after measuring the specific simulator. The effective
value is stored in `runtime.cpu_threads`.

### Process-level GPU parallelism

Task, protocol, seed, and checkpoint cells are independent. The reliable
multi-GPU strategy is one evaluator process per free GPU. This preserves each
manifest and avoids distributed synchronization inside the short two-replan
LeWM control loop.

### Other defaults

- Video is disabled unless `--video-dir` is provided.
- Official checkpoints are loaded once and put in inference mode.
- Current and goal embeddings are cached by the installed LeWM runtime across
  CEM iterations.
- Dataset fingerprints are checked before environments are created.
- Environment cleanup is guaranteed for setup, policy, and rollout failures.

## FAST training input

FAST moves repeated image decoding and row-store lookup into a one-time
conversion. The runtime reader then slices raw memory-mapped arrays.

The corrected implementation differs from the early experimental loader in
six important ways:

1. `_load_slice` applies `self.transform`.
2. Action rows are not observation-strided, preserving the complete action
   chunk expected by the base dataset reshaper.
3. Schema, completion, shape, dtype, file size, and episode boundaries are
   validated before reading.
4. Resume refuses a completed, malformed, differently shaped, or differently
   sourced snapshot.
5. `audit_fast_dataset.py` checks full numeric columns and fixed seeded clips
   against the authoritative reader.
6. Conversion uses a source reader's batched fetch when available and one
   vectorized memmap assignment per column, with a generic reader fallback.

The four-task audit checked 258 fixed and seeded clips per task, every complete
numeric column, episode lengths, and episode offsets. All four FAST snapshots
matched their authoritative training readers exactly. Machine-readable audit
records are under [`benchmarks/`](benchmarks/).

### Four-task steady-state loader benchmark

The formal 2026-07-23 run used one H200-2 node and local storage. Conversion
writes were already complete; tasks ran serially, and Source/FAST order
alternated within each task. Each backend ran in a fresh subprocess with
`num_steps=4`, `frameskip=5`, batch size 32, zero workers, three in-process
warm-up batches, 30 measured batches, and no image transform or model compute.
Cube, Reacher, and TwoRoom discard one complete fixed-index pair to prime the
OS page cache, then measure five fixed-seed pairs. A confirmatory PushT rerun
uses two discarded priming pairs and ten measured pairs. Speedup is each task's
median paired FAST/Source ratio.

| Task | Source reader | Source samples/s | FAST samples/s | Paired speedup | Source / FAST CV |
|---|---|---:|---:|---:|---:|
| PushT | Lance | 672.3 | 3812.0 | **5.79x** | 0.074 / 0.078 |
| Cube | HDF5 | 119.5 | 4426.5 | **37.72x** | 0.029 / 0.069 |
| Reacher | HDF5 | 143.0 | 4362.1 | **30.77x** | 0.007 / 0.068 |
| TwoRoom | HDF5 | 279.2 | 4291.4 | **15.15x** | 0.025 / 0.020 |

Across the four paired task speedups, the **geometric mean is 17.87x** and the
arithmetic mean is 22.36x. CLEAR-LeWM uses the geometric mean as its headline
aggregate because ratios combine multiplicatively across heterogeneous
benchmarks. The larger HDF5 ratios measure removal of random row access and
repeated image decoding. They are loader-only results, not claims of 15-38x
end-to-end training speedup. Snapshot conversion time is also excluded;
conversion is a one-time preprocessing cost whose throughput is sensitive to
concurrent writes and storage bandwidth.

### Historical PushT observations

An earlier loader-only benchmark on H20-3 used the same clip geometry and batch
size but three alternating rounds without the formal fixed-index priming pass.
Median throughput was 503.7 samples/s for Lance and 4660.6 samples/s for FAST,
or 9.25x. The individual Lance rates were 503.7, 584.3, and 471.4 samples/s;
FAST produced 3990.4, 4660.6, and 5429.4 samples/s. This remains a transparent
node-specific historical result, not the four-task headline.

A historical H200 training observation at batch 128 and 16 loader workers was
11.0 steps/s for FAST and 6.1 steps/s for Lance, or about 1.8x. Treat this as an
operational observation: storage, page cache, compression, workers, transforms,
and GPU compute can move the ratio substantially. The smaller end-to-end gain
is expected because model compute is unchanged.

FAST should live on local NVMe. The PushT snapshot is approximately 328 GiB;
placing it on network storage can erase its latency advantage. Evaluation stays
on canonical HDF5 so manifest row identities do not depend on a training format.

## Batched CEM throughput mode

The upstream CEM solver exposes `batch_size`, but canonical LeWM's cost function
omits a sample-axis `unsqueeze`, so values above one fail broadcasting. When
`--solver-batch-size` is greater than one, CLEAR-LeWM patches that canonical
criterion and records `runtime.batched_lewm_criterion_patch=true`.

The following sweep used one H20-3e per row, the official PushT checkpoint,
PushT Strict's fixed 100-pair manifest, and `300 x 30` CEM. Time is the sum of
the two printed CEM solves required by the 50-step evaluation.

| Solver batch | CEM time | Speedup | SR |
|---:|---:|---:|---:|
| 1, official reference | 112.8 s | 1.00x | 42% |
| 2 | 84.7 s | 1.33x | 50% |
| 4 | 79.5 s | 1.42x | 55% |
| 8 | 77.3 s | 1.46x | 50% |
| 16 | 75.6 s | 1.49x | 46% |
| 32 | 74.6 s | 1.51x | 46% |
| 100 | 73.2 s | 1.54x | 49% |

The plateau begins around batch 16. Batch 16 is therefore a reasonable
development setting, but it is not reference-equivalent. CEM draws from one
generator inside its batch loop; changing batch size reorders the random draws
assigned to each environment. Small batched floating-point differences can
further change elite selection. Use batch 1 for comparable tables.

### Four-task trajectory-equivalence check

A second controlled check used one LeWM-compatible custom checkpoint per task,
the fixed Strict seed-0 manifests, 100 episodes per task, and the same `300 x 30`
CEM budget. Each batch-1/batch-16 pair ran on the same H20-3e and differed only
in solver batch size. Times below cover evaluation after setup, not only the two
CEM calls.

| Task | Batch 1 SR | Batch 16 SR | Delta | Outcome flips | Evaluation speedup |
|---|---:|---:|---:|---:|---:|
| PushT | 49% | 41% | -8pp | 28/100 | 1.52x |
| Cube | 50% | 48% | -2pp | 10/100 | 1.47x |
| Reacher | 25% | 25% | 0pp | 40/100 | 1.47x |
| TwoRoom | 19% | 14% | -5pp | 21/100 | 1.62x |
| Macro / total | 35.75% | 32.00% | -3.75pp | 99/400 | 1.51x |

The Reacher totals are the sharpest warning: 20 successes became failures and
20 failures became successes, so equal aggregate SR concealed 40 changed
outcomes. Across all tasks, 57 successes became failures and 42 failures became
successes. The net SR shift can therefore look modest while the underlying
planner trajectories are substantially different.

This follows directly from the upstream loop order. One shared Torch generator
is consumed inside the environment-batch loop; changing batch size changes which
environment and CEM iteration receives each random draw. Batched kernels can
then add small floating-point differences before top-k elite selection. Batch 16
is useful for rapid screening, but it defines a different stochastic planner
trace and must never replace batch 1 in reference or model-selection tables.
The machine-readable aggregate is
[`benchmarks/cem_batch_four_task_20260722.json`](benchmarks/cem_batch_four_task_20260722.json).

`--matmul-precision high` or `medium` is available for cross-hardware tests and
is recorded in `runtime.float32_matmul_precision`, but it is not recommended on
the measured H20-3e setup. `high` took 116.9 s at batch 1 and 81.1 s at batch
16, both slower than the corresponding 112.8 s and 75.6 s default-precision
runs; SR also changed to 44% and 50%. Reduced `--num-samples`, `--n-steps`, or
`--topk` changes the planner budget directly and must be reported as a separate
algorithmic setting.

## Remaining bottlenecks

1. CEM world-model calls dominate official model evaluation. Batching improves
   utilization but does not reduce total candidate computation.
2. The synchronous Python environment pool renders every active environment.
   Parallel task cells across GPUs are more effective than distributed control
   inside one 100-pair rollout.
3. Normalizer fitting scans small state/action columns once per process. A
   fingerprinted statistics cache could reduce startup, but it is not dominant
   in `300 x 30` runs and is intentionally not added without a cache audit.
4. `torch.compile` has substantial first-call cost while standard LeWM performs
   only two planning calls per episode batch; it is not enabled by default.
5. Saving videos adds frame copies and encoding. It should be reserved for
   qualitative subsets rather than SR sweeps.
