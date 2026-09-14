# INTACT Direct and Guarded A: CLEAR-LeWM v0.8 on RTX 4090

Three public [INTACT](https://huggingface.co/INTACT-JEPA/INTACT) checkpoint families were evaluated with their matching Direct actor (query target, receding horizon 5, no CEM or sampled search) and Guarded A (actor-guided search, 128 samples x 3 iterations). The [task-specific E1 history](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT), [task-specific E1 no-previous-action](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-no-previous-action), and [unified E5](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-unified) families each have four task checkpoints for training seeds 0, 42, and 3072 (36 checkpoint shards in total). These are CLEAR Moderate/Strict scores, **not Official Direct** scores; the pure-CEM reference runs elsewhere in this directory use a different inference method.

The evaluation uses pinned v0.8 manifests, RTX 4090, evaluation seeds 0/1/42, and 100 episodes per checkpoint/task/mode/evaluation seed **per inference method** (216 Direct runs and 216 Guarded A runs, 43,200 episodes). For each training seed, the three evaluation seeds are pooled into 300 episodes. Table entries are the mean and sample standard deviation **across the three training seeds**, including for the equal-weight four-task macro. The history-enabled actor is initialized only from actions before the sampled start and updated from executed policy actions thereafter; no current or future expert action is available. The no-previous-action actor ignores its history channel.

## Direct (no sampled search)

| Checkpoint | Mode | PushT | Cube | Reacher | TwoRoom | Macro SR |
|---|---|---:|---:|---:|---:|---:|
| Task-specific E1 (history) | Moderate | 89.56 +/- 1.17% | 100.00 +/- 0.00% | 98.22 +/- 0.69% | 96.89 +/- 3.47% | 96.17 +/- 0.60% |
|  | Strict | 79.67 +/- 0.33% | 87.33 +/- 1.20% | 97.67 +/- 0.58% | 93.00 +/- 5.17% | 89.42 +/- 0.96% |
| Task-specific E1 (no previous action) | Moderate | 89.56 +/- 1.84% | 98.78 +/- 0.38% | 93.67 +/- 1.20% | 98.11 +/- 1.50% | 95.03 +/- 0.69% |
|  | Strict | 76.33 +/- 2.52% | 84.33 +/- 1.45% | 96.11 +/- 0.77% | 96.67 +/- 0.88% | 88.36 +/- 0.46% |
| Unified E5 | Moderate | 89.00 +/- 1.15% | 99.89 +/- 0.19% | 98.22 +/- 0.84% | 79.44 +/- 5.19% | 91.64 +/- 0.88% |
|  | Strict | 76.00 +/- 1.53% | 91.67 +/- 1.45% | 98.78 +/- 0.51% | 53.00 +/- 10.27% | 79.86 +/- 2.96% |

## Guarded A (128 x 3 actor-guided search)

| Checkpoint | Mode | PushT | Cube | Reacher | TwoRoom | Macro SR |
|---|---|---:|---:|---:|---:|---:|
| Task-specific E1 (history) | Moderate | 93.89 +/- 1.84% | 99.22 +/- 0.19% | 95.89 +/- 0.69% | 96.78 +/- 3.29% | 96.44 +/- 0.50% |
|  | Strict | 85.22 +/- 1.26% | 86.78 +/- 0.51% | 96.33 +/- 1.20% | 94.44 +/- 4.30% | 90.69 +/- 1.00% |
| Task-specific E1 (no previous action) | Moderate | 94.22 +/- 0.96% | 98.56 +/- 0.38% | 94.00 +/- 1.67% | 98.11 +/- 1.68% | 96.22 +/- 0.32% |
|  | Strict | 82.11 +/- 1.35% | 83.44 +/- 1.84% | 96.89 +/- 1.50% | 96.00 +/- 1.86% | 89.61 +/- 0.84% |
| Unified E5 | Moderate | 88.22 +/- 3.37% | 98.44 +/- 0.96% | 97.78 +/- 0.38% | 78.00 +/- 7.33% | 90.61 +/- 1.55% |
|  | Strict | 70.33 +/- 7.80% | 91.11 +/- 2.41% | 98.33 +/- 0.33% | 73.11 +/- 7.40% | 83.22 +/- 1.13% |

## Direct training-seed means (%)

Each value pools three 100-episode evaluation seeds for one checkpoint.

| Checkpoint | Task | Mode | Train 0 | Train 42 | Train 3072 |
|---|---|---|---:|---:|---:|
| Task-specific E1 (history) | PushT | Moderate | 90.67 | 89.67 | 88.33 |
|  | Cube | Moderate | 100.00 | 100.00 | 100.00 |
|  | Reacher | Moderate | 99.00 | 98.00 | 97.67 |
|  | TwoRoom | Moderate | 93.00 | 99.67 | 98.00 |
|  | PushT | Strict | 79.33 | 80.00 | 79.67 |
|  | Cube | Strict | 88.67 | 86.33 | 87.00 |
|  | Reacher | Strict | 98.33 | 97.33 | 97.33 |
|  | TwoRoom | Strict | 87.67 | 98.00 | 93.33 |
| Task-specific E1 (no previous action) | PushT | Moderate | 89.67 | 91.33 | 87.67 |
|  | Cube | Moderate | 98.33 | 99.00 | 99.00 |
|  | Reacher | Moderate | 94.67 | 94.00 | 92.33 |
|  | TwoRoom | Moderate | 99.67 | 96.67 | 98.00 |
|  | PushT | Strict | 74.00 | 79.00 | 76.00 |
|  | Cube | Strict | 85.33 | 85.00 | 82.67 |
|  | Reacher | Strict | 97.00 | 95.67 | 95.67 |
|  | TwoRoom | Strict | 97.00 | 95.67 | 97.33 |
| Unified E5 | PushT | Moderate | 88.33 | 88.33 | 90.33 |
|  | Cube | Moderate | 100.00 | 100.00 | 99.67 |
|  | Reacher | Moderate | 97.33 | 98.33 | 99.00 |
|  | TwoRoom | Moderate | 84.33 | 80.00 | 74.00 |
|  | PushT | Strict | 76.33 | 77.33 | 74.33 |
|  | Cube | Strict | 92.33 | 92.67 | 90.00 |
|  | Reacher | Strict | 98.33 | 98.67 | 99.33 |
|  | TwoRoom | Strict | 64.00 | 51.33 | 43.67 |

## Guarded A training-seed means (%)

| Checkpoint | Task | Mode | Train 0 | Train 42 | Train 3072 |
|---|---|---|---:|---:|---:|
| Task-specific E1 (history) | PushT | Moderate | 96.00 | 92.67 | 93.00 |
|  | Cube | Moderate | 99.33 | 99.33 | 99.00 |
|  | Reacher | Moderate | 95.33 | 96.67 | 95.67 |
|  | TwoRoom | Moderate | 93.00 | 99.00 | 98.33 |
|  | PushT | Strict | 86.67 | 84.67 | 84.33 |
|  | Cube | Strict | 87.33 | 86.67 | 86.33 |
|  | Reacher | Strict | 95.33 | 97.67 | 96.00 |
|  | TwoRoom | Strict | 89.67 | 98.00 | 95.67 |
| Task-specific E1 (no previous action) | PushT | Moderate | 93.67 | 95.33 | 93.67 |
|  | Cube | Moderate | 99.00 | 98.33 | 98.33 |
|  | Reacher | Moderate | 94.00 | 92.33 | 95.67 |
|  | TwoRoom | Moderate | 99.67 | 98.33 | 96.33 |
|  | PushT | Strict | 81.33 | 83.67 | 81.33 |
|  | Cube | Strict | 84.33 | 84.67 | 81.33 |
|  | Reacher | Strict | 98.33 | 95.33 | 97.00 |
|  | TwoRoom | Strict | 98.00 | 94.33 | 95.67 |
| Unified E5 | PushT | Moderate | 84.33 | 90.33 | 90.00 |
|  | Cube | Moderate | 99.00 | 99.00 | 97.33 |
|  | Reacher | Moderate | 98.00 | 98.00 | 97.33 |
|  | TwoRoom | Moderate | 85.33 | 78.00 | 70.67 |
|  | PushT | Strict | 61.33 | 75.00 | 74.67 |
|  | Cube | Strict | 92.33 | 92.67 | 88.33 |
|  | Reacher | Strict | 98.00 | 98.67 | 98.33 |
|  | TwoRoom | Strict | 81.33 | 71.00 | 67.00 |

These are independently evaluated aggregate results; unlike the existing CEM reference bundles under `runs/`, episode-level INTACT Direct and Guarded A traces are not published in this repository.
