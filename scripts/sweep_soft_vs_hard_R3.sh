#!/usr/bin/env bash
# §6.2 Soft-vs-hard ablation, Stage 2: amplified-capacity reference R''.
#
# R'' = R' + --r_star 32 + --gumbel_alpha 0.3
#   - r=32 gives the assignment matrix more capacity to differ between soft
#     and hard variants (hard's argmax becomes more brittle in 32 classes).
#   - Gumbel noise α=0.3 helps the soft variant explore richer partitions
#     during training; hard via STE is largely invariant to the noise scale
#     (argmax is invariant under monotone transforms).
#
# Submitted in parallel with sweep_soft_vs_hard.sh; cluster queues.
# 8 channels x 2 variants = 16 jobs.

set -euo pipefail

declare -a JOBS
JOBS+=("CalGeo-AirQuality-pm25"    64)
JOBS+=("CalGeo-Solar-ghi"          64)
JOBS+=("CalGeo-Weather-tmax"       64)
JOBS+=("CalGeo-Weather-prcp"       64)
JOBS+=("Mobility-CA-outdoor"       64)
JOBS+=("Mobility-CA-essential"     16)
JOBS+=("Mobility-CA-indoor"        16)
JOBS+=("Mobility-CA-food"           8)

R3_FLAGS="--use_distance_anchor --distance_alpha 1.0 --use_coord_embed --use_laplacian_smooth --laplacian_lambda 0.05 --use_graph_prop --use_adaptive_adj --adaptive_adj_dim 32 --scale_mode topk --topk_within 8 --r_star 32 --gumbel_alpha 0.3"
COMMON=(--report_raw_metrics --train_epochs 100 --patience 15 --loss L1)

MANIFEST="${IJGIS_ROOT:-../ijgis}/v4/soft_vs_hard_R3_manifest.tsv"
mkdir -p "$(dirname "$MANIFEST")"
echo -e "channel\tvariant\tjob_id\tbatch_size\tflags\tsubmitted_at" > "$MANIFEST"

submit() {
  local chan="$1"; local bs="$2"; local var="$3"
  local short_chan="${chan##*-}"
  local des="ablation_${short_chan}_R3_${var}"
  local extra=""
  if [ "$var" = "hardW" ]; then
    extra="--hard_assignment"
  fi
  local out
  out=$(bash scripts/submit_ray_spatiallca.sh --model RegionFormer "$chan" \
        --des "$des" \
        "${COMMON[@]}" $R3_FLAGS --batch_size "$bs" $extra 2>&1)
  local job
  job=$(echo "$out" | grep -oE 'raysubmit_[A-Za-z0-9]+' | head -1)
  echo "  [$short_chan/$var bs=$bs] -> $job"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$chan" "$var" "$job" "$bs" "$R3_FLAGS $extra" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
}

echo "Submitting Stage 2 (r=32 + Gumbel) — 8 channels x 2 variants = 16 jobs..."
for ((i=0; i<${#JOBS[@]}; i+=2)); do
  chan="${JOBS[i]}"
  bs="${JOBS[i+1]}"
  submit "$chan" "$bs" "softW"
  submit "$chan" "$bs" "hardW"
done

echo
echo "Submitted 16 R3 jobs. Manifest: $MANIFEST"
