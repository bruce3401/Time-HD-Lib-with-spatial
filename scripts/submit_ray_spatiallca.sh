#!/usr/bin/env bash
# Submit a SpatialLCA experiment to the the user-specified Ray cluster.
#
# Usage:
#   bash scripts/submit_ray_spatiallca.sh <DATA> [extra_args...]
#
# Example:
#
# Notes:
#   * Each call grabs 1 GPU. Run twice in parallel for an A/B ablation.
#   * /data inside the container = /opt/data on host (ray-head mount).
#   * Datasets at /data/timehd/, checkpoints at /data/checkpoints-timehd/.
#   * Fresh runs upload the working dir each time (~5 MB after excludes).

set -euo pipefail
RAY_ADDR="${RAY_ADDR:?Set RAY_ADDR env var to your Ray cluster URL, e.g. http://localhost:8265}"
# Either: submit_ray_spatiallca.sh <DATA> [extra_args]              (--model SpatialLCA, default)
#     or: submit_ray_spatiallca.sh --model <NAME> <DATA> [extra_args]   (override model)
MODEL="SpatialLCA"
if [ "${1:-}" = "--model" ]; then
  MODEL="${2:?--model needs a value}"
  shift 2
fi
DATA="${1:?usage: submit_ray_spatiallca.sh [--model NAME] <DATA> [extra_args]}"
shift

# Pick root_path by dataset family unless caller overrides via env.
case "$DATA" in
  Mobility-CA-*) ROOT_PATH="${ROOT_PATH:-/data/timehd/Mobility_CA/}" ;;
  *)             ROOT_PATH="${ROOT_PATH:-/data/timehd/}" ;;
esac

EXTRA_ARGS=("$@")

# Pull WANDB_API_KEY (and optional WANDB_PROJECT/MODE) from the shared .env so
# we don't bake secrets into git. WANDB_PROJECT defaults to spatialscale here
# regardless of the .env value, since SpatialScale-IJGIS is what this script
# is for.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for env_file in "${SCRIPT_DIR}/../.env" "${SCRIPT_DIR}/../../.env"; do
  if [ -f "$env_file" ]; then set -a; source "$env_file"; set +a; break; fi
done
WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-spatialscale}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

RUNTIME_ENV=$(python3 -c '
import json, os
env = {
  "pip": [
    "--extra-index-url https://download.pytorch.org/whl/nightly/cu128",
    "--pre",
    "torch",
    "numpy>=1.24.0", "pandas>=2.0.0", "scikit-learn>=1.3.0",
    "scipy>=1.10.0", "statsmodels>=0.14.0",
    "accelerate>=0.27.0", "einops>=0.7.0", "tqdm>=4.65.0",
    "matplotlib>=3.7.0", "tables>=3.9.0", "datasets>=2.14.0",
    "arch>=6.3.0", "wandb>=0.16.0"
  ],
  "env_vars": {
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
    "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "spatialscale"),
    "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
  },
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
  -- python run.py \
       --model "$MODEL" \
       --data "$DATA" \
       --root_path "$ROOT_PATH" \
       --checkpoints /data/checkpoints-timehd/ \
       "${EXTRA_ARGS[@]}"
