#!/usr/bin/env bash
# §6.2 Soft-vs-hard assignment ablation, extended.
# Replaces sweep_hard_assignment.sh and the soft cells of sweep_ablation.sh
# with a single unified sweep: 8 channels x 2 variants = 16 jobs.
#
# R' config (uniform, used for both soft and hard variants):
#   --use_distance_anchor --distance_alpha 1.0 --use_coord_embed
#   --use_laplacian_smooth --laplacian_lambda 0.05 --use_graph_prop
#   --use_adaptive_adj --adaptive_adj_dim 32
#   --scale_mode topk --topk_within 8
#   --loss L1
# (Drops --use_dense_attn vs the old R, since dense_attn OOMs on large
# mobility channels; topk is the §6.1 scaling story anyway.)
#
# Channels span all 4 CalST-Bench domains:
#   AQ:        pm25
#   Solar:     ghi
#   Weather:   tmax, prcp  (kept to honestly show the autocorr-dominated tie)
#   Mobility:  outdoor, essential, indoor, food

set -euo pipefail

declare -a JOBS
# (channel, batch_size) pairs — large N gets smaller batch.
JOBS+=("CalGeo-AirQuality-pm25"    64)
JOBS+=("CalGeo-Solar-ghi"          64)
JOBS+=("CalGeo-Weather-tmax"       64)
JOBS+=("CalGeo-Weather-prcp"       64)
JOBS+=("Mobility-CA-outdoor"       64)
JOBS+=("Mobility-CA-essential"     16)
JOBS+=("Mobility-CA-indoor"        16)
JOBS+=("Mobility-CA-food"           8)

R_FLAGS="--use_distance_anchor --distance_alpha 1.0 --use_coord_embed --use_laplacian_smooth --laplacian_lambda 0.05 --use_graph_prop --use_adaptive_adj --adaptive_adj_dim 32 --scale_mode topk --topk_within 8"
COMMON=(--report_raw_metrics --train_epochs 100 --patience 15 --loss L1)

MANIFEST="${IJGIS_ROOT:-../ijgis}/v4/soft_vs_hard_manifest.tsv"
mkdir -p "$(dirname "$MANIFEST")"
echo -e "channel\tvariant\tjob_id\tbatch_size\tflags\tsubmitted_at" > "$MANIFEST"

submit() {
  local chan="$1"; local bs="$2"; local var="$3"
  local short_chan="${chan##*-}"
  local des="ablation_${short_chan}_R2_${var}"
  local extra=""
  if [ "$var" = "hardW" ]; then
    extra="--hard_assignment"
  fi
  local out
  out=$(bash scripts/submit_ray_spatiallca.sh --model RegionFormer "$chan" \
        --des "$des" \
        "${COMMON[@]}" $R_FLAGS --batch_size "$bs" $extra 2>&1)
  local job
  job=$(echo "$out" | grep -oE 'raysubmit_[A-Za-z0-9]+' | head -1)
  echo "  [$short_chan/$var bs=$bs] -> $job"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$chan" "$var" "$job" "$bs" "$R_FLAGS $extra" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
}

echo "Submitting 8 channels x 2 variants = 16 jobs..."
for ((i=0; i<${#JOBS[@]}; i+=2)); do
  chan="${JOBS[i]}"
  bs="${JOBS[i+1]}"
  submit "$chan" "$bs" "softW"
  submit "$chan" "$bs" "hardW"
done

echo
echo "Submitted 16 jobs. Manifest: $MANIFEST"
