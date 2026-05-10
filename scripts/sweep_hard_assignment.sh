#!/usr/bin/env bash
# §6.2 Soft-vs-hard assignment ablation.
# Same uniform R config as sweep_ablation.sh, but with --hard_assignment.
# Reuses the existing R results as the soft baseline; this sweep produces the
# 5 hard-assignment cells.
#
# 5 channels × 1 variant = 5 jobs.

set -euo pipefail

CHANNELS=(
  "CalGeo-AirQuality-pm25"
  "CalGeo-Solar-ghi"
  "CalGeo-Weather-tmax"
  "CalGeo-Weather-prcp"
  "Mobility-CA-outdoor"
)

# R = uniform reference (matches sweep_ablation.sh R)
R_FLAGS="--use_distance_anchor --distance_alpha 1.0 --use_coord_embed --use_laplacian_smooth --laplacian_lambda 0.05 --use_graph_prop --use_dense_attn --use_adaptive_adj --adaptive_adj_dim 32"
COMMON=(--report_raw_metrics --train_epochs 100 --patience 15 --batch_size 64 --loss L1)

MANIFEST="${IJGIS_ROOT:-../ijgis}/v4/hard_assignment_manifest.tsv"
mkdir -p "$(dirname "$MANIFEST")"
echo -e "channel\tvariant\tjob_id\tflags\tsubmitted_at" > "$MANIFEST"

submit() {
  local chan="$1"
  local short_chan="${chan##*-}"
  local des="ablation_${short_chan}_hardW"
  local out
  out=$(bash scripts/submit_ray_spatiallca.sh --model RegionFormer "$chan" \
        --des "$des" \
        "${COMMON[@]}" $R_FLAGS --hard_assignment 2>&1)
  local job
  job=$(echo "$out" | grep -oE 'raysubmit_[A-Za-z0-9]+' | head -1)
  echo "  [$short_chan/hardW] -> $job"
  printf "%s\thardW\t%s\t%s\t%s\n" "$chan" "$job" "$R_FLAGS --hard_assignment" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
}

echo "Submitting 5 hard-assignment ablation jobs..."
for chan in "${CHANNELS[@]}"; do
  submit "$chan"
done

echo
echo "Submitted 5 hard-assignment jobs. Manifest: $MANIFEST"
