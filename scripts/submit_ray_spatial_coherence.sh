#!/usr/bin/env bash
# Submit F4 spatial-coherence figure generation on cluster.
# Loads a trained ablation_R checkpoint, extracts assignment matrix W,
# plots California map coloured by argmax centroid.
#
# Usage:  bash scripts/submit_ray_spatial_coherence.sh <channel>
#   channel ∈ {pm25, outdoor}

set -euo pipefail
CHAN="${1:?usage: submit_ray_spatial_coherence.sh <channel>}"
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
  -- python scripts/plot_spatial_coherence.py --channel "$CHAN"
