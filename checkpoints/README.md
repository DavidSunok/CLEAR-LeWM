# Official checkpoint registry

The repository records immutable source revisions and SHA-256 identities, not
large binary weights. Prepare all four official LeWM checkpoints with:

```bash
python scripts/prepare_official_checkpoints.py --cache-dir "$STABLEWM_HOME"
```

The script verifies the downloaded source hash, reconstructs the canonical
architecture, strictly loads all 303 tensors, and writes a runtime hash sidecar.
The release registry is [`official-v0.3.json`](official-v0.3.json).
