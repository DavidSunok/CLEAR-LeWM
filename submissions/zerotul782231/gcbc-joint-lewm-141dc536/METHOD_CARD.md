# GCBC Joint LeWM

Contact: `@zerotul782231`

## Scope and classification

Task-specific joint goal-conditioned/world-model checkpoints, evaluated by pure CEM over the serialized forward world model.

These are **related public checkpoint evaluations**. They are not claimed to be the
DINO-WM, DINO-WM+prop, PLDM, CBC, GCIQL, or GCIVL releases. The public
checkpoint collection contains related task-specific serialized models but
does not provide an exact baseline mapping or the newer runtime that produced
their auxiliary action heads. Reacher is intentionally absent because no
accepted checkpoint for this family was found.

## Canonical CLEAR-LeWM results

- evaluator: CLEAR-LeWM `0.5.1`
- benchmark protocol version: `v0.5`
- manifests: checked-in `moderate-seed42-n100.json` and
  `strict-seed42-n100.json`
- policy/CEM seed: `42`
- episodes: `100` per task/protocol
- inference: CEM, action-head warm start disabled
- budget: 300 samples, 30 iterations, top-k 30, solver batch size 1
- training-data track: `standard-data`

| Task | Protocol | SR (%) | Random (%) | Excess (pp) |
|---|---|---|---|---|
| tworoom | moderate | 15.00 | 6.00 | 9.00 |
| tworoom | strict | 9.00 | 0.00 | 9.00 |
| pusht | moderate | 9.00 | 3.00 | 6.00 |
| pusht | strict | 9.00 | 7.00 | 2.00 |
| cube | moderate | 16.00 | 15.00 | 1.00 |
| cube | strict | 3.00 | 8.00 | -5.00 |

All result JSON files retain the full 100-episode Boolean trace, paired
canonical random rate, strict state-dict audit, runtime target audit,
environment fingerprints, and TwoRoom route audit where applicable.

## Checkpoints

Repository: [https://huggingface.co/MinghaoFu/lewm-official-ckpts](https://huggingface.co/MinghaoFu/lewm-official-ckpts)

Immutable revision: `141dc536b247898164a7422e3bbb7514429529b6`

- `tworoom`: [tworoom_gc_e2e/tworoom_gc_e2e__weights_epoch_40.pt](https://huggingface.co/MinghaoFu/lewm-official-ckpts/blob/141dc536b247898164a7422e3bbb7514429529b6/checkpoints/tworoom_gc_e2e/weights_epoch_40.pt), SHA256 `d30b2eddbfd17ce9281d55de54e15a268e973fa0fb51273d074b0ff524b70ccf`
- `pusht`: [pusht_gc_e2e/pusht_gc_e2e__weights_epoch_10.pt](https://huggingface.co/MinghaoFu/lewm-official-ckpts/blob/141dc536b247898164a7422e3bbb7514429529b6/checkpoints/pusht_gc_e2e/weights_epoch_10.pt), SHA256 `fa6c59823857c0aa6277880b1523ec34c7fe8dbbe6344bbbf18bd651d7f28fa5`
- `cube`: [cube_gc_e2e/cube_gc_e2e__weights_epoch_60.pt](https://huggingface.co/MinghaoFu/lewm-official-ckpts/blob/141dc536b247898164a7422e3bbb7514429529b6/checkpoints/cube_gc_e2e/weights_epoch_60.pt), SHA256 `41c4f61a7e424d15a52bdd8c06cb96009560447c41a153c699cd47f4f5ab3e1f`

The publisher repository did not provide a checkpoint model-card license at
the audited revision. This submission therefore requests `self-reported`
verification and records the method license as unspecified by the publisher.
Dataset names in the frozen training configs correspond to the public LeWM
task datasets; their dataset-specific upstream terms continue to apply.

## Runtime compatibility adapter

`runtime/adapter.py.txt` (SHA256 `5b443b94ef18ad7fec4a34a6f12b3d3f2523f75cd2bdf4dd1914764c39789619`) is the byte-exact audited
adapter source. Copy it to a temporary runtime directory as `adapter.py` before
evaluation. It reconstructs the token-type
embeddings, horizon modulator, and auxiliary inverse-action modules present in
the serialized state dicts. Every evaluated checkpoint loaded with zero
missing and zero unexpected keys. CEM uses the forward world model; the
auxiliary action head is instantiated for strict load compatibility and is not
used to propose actions.

`supporting/configs/` contains portable adapted configs derived from the
publisher's source configs. For DINO-like models the evaluated config used a
local pinned copy of `facebook/dinov2-small`; the portable config uses its
public model identifier. Strict state-dict loading overwrites the instantiated
encoder parameters. Both the evaluated and portable config hashes are recorded
in `supporting/checkpoint-provenance.json`.

## Provenance-only result normalization

The original evaluator outputs are preserved by SHA256 in
`supporting/result-transformations.json`. Submission packaging made only two
metadata changes:

1. added `checkpoint.source` from the immutable checkpoint audit;
2. for Moderate only, serialized `protocol` exactly as stored in the canonical
   manifest, removing three equal-valued dataclass defaults emitted by the
   v0.5.1 runner.

Episode traces, metrics, dataset identity, seeds, solver settings, runtime
fingerprints, and topology records were not changed. The official submission
validator checks the normalized result hashes declared in `submission.json`.

## Supplementary five-seed evidence

The supplementary files contain 5-seed robustness statistics and seed-level
records. They are not declared in `submission.json` because they do not satisfy
the current canonical seed42 contract:

- Moderate five-seed archive SHA256 `40bc46f1fd73e99d3f5a887652d8a65abb42ac66734bd7bf6d3809306ab97dba`,
  policy seeds 0-4, evaluator 0.5.0;
- Strict five-seed archive SHA256 `82a398f640874592953808c8f3fe7421af22f2d7ceae6b85fd334b3a113dec06`,
  five random 31-bit seeds and separately
  generated manifests.

| Task | Protocol | 5-seed mean (%) | Sample SD (%) | Mean random (%) | Mean excess (pp) |
|---|---|---|---|---|---|
| cube | moderate | 16.40 | 5.18 |  |  |
| cube | strict | 2.80 | 1.64 | 4.40 | -1.60 |
| pusht | moderate | 5.60 | 2.51 |  |  |
| pusht | strict | 8.00 | 2.55 | 3.60 | 4.40 |
| tworoom | moderate | 12.60 | 1.52 |  |  |
| tworoom | strict | 8.40 | 2.07 | 2.60 | 5.80 |

Sample variance and sample standard deviation use the `n-1` denominator.
`supplementary/5seed-metrics.csv` retains each seed-level result digest and its
manifest digest; none is represented as a canonical leaderboard result.

## Reproduction outline

From a CLEAR-LeWM `v0.5.1` checkout with its LeWM submodule and `[lewm]`
dependencies installed:

```bash
export STABLEWM_HOME=/path/to/stablewm-cache
export METHOD_BUNDLE=/path/to/this/submission/bundle
export METHOD_RUNTIME=/tmp/clear-lewm-checkpoint-runtime
mkdir -p "$METHOD_RUNTIME"
cp "$METHOD_BUNDLE/runtime/adapter.py.txt" "$METHOD_RUNTIME/adapter.py"

# Download each checkpoint listed above at the frozen revision, verify SHA256,
# install it as $STABLEWM_HOME/checkpoints/<policy-id>/weights.pt, and copy the
# matching portable config to config.json.

clear-lewm evaluate \
  --manifest manifests/v0.5/<task>/<protocol>-seed42-n100.json \
  --policy gcbc-joint-lewm/<task> \
  --policy-label gcbc-joint-lewm-141dc536-pure-cem \
  --cache-dir "$STABLEWM_HOME" \
  --runtime-dir "$METHOD_RUNTIME" \
  --policy-seed 42 \
  --num-samples 300 --n-steps 30 --topk 30 \
  --actor-warmstart off --inference-mode cem \
  --solver-batch-size 1 --strict-checkpoint \
  --random-results results/v0.5/<task>-<protocol>-random-seed42-n100.json \
  --output /path/to/result.json
```

The external datasets and checkpoint binaries are deliberately not committed.
