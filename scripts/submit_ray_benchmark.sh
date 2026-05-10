#!/usr/bin/env bash
# Submit the LRA scaling benchmark as a one-shot Ray job.
# Pulls torch via Ray's runtime-env pip block (matches submit_ray_spatiallca.sh).

set -euo pipefail
RAY_ADDR="${RAY_ADDR:?Set RAY_ADDR env var to your Ray cluster URL, e.g. http://localhost:8265}"

RUNTIME_ENV=$(python3 -c '
import json
env = {
  "pip": [
    "--extra-index-url https://download.pytorch.org/whl/nightly/cu128",
    "--pre",
    "torch",
    "numpy>=1.24.0",
    "einops>=0.7.0",
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
  -- python scripts/benchmark_lra_scaling.py
