#!/usr/bin/env bash
# Submit case1 W-extraction across all 8 R2_softW channels.
set -euo pipefail
RAY_ADDR="${RAY_ADDR:?Set RAY_ADDR env var to your Ray cluster URL, e.g. http://localhost:8265}"

RUNTIME_ENV=$(python3 -c '
import json
env = {
  "pip": [
    "--extra-index-url https://download.pytorch.org/whl/nightly/cu128",
    "--pre",
    "torch",
    "numpy>=1.24.0", "pandas>=2.0.0",
    "matplotlib>=3.7.0", "einops>=0.7.0",
  ],
  "excludes": ["dataset", "results", "checkpoints", "wandb", "logs",
               ".git", "viz", "*.pdf", "*.npz", "*.pt", "*.ckpt",
               "__pycache__", "*.egg-info"]
}
print(json.dumps(env))
')

ray job submit \
  --address "$RAY_ADDR" \
  --working-dir . \
  --no-wait \
  --runtime-env-json "$RUNTIME_ENV" \
  --entrypoint-num-gpus 1 \
  -- python scripts/case_study/extract_W_all_channels.py
