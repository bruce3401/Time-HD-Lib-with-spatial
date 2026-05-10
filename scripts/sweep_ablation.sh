#!/usr/bin/env bash
# §6 Ablation sweep — uniform RegionFormer reference R, leave-one-out across
# 2 priors + 2 modes, on 5 representative channels.
#
# Reference R (all components on):
#   --use_distance_anchor --distance_alpha 1.0 --use_coord_embed
#   --use_laplacian_smooth --laplacian_lambda 0.05 --use_graph_prop
#   --use_dense_attn
#   --use_adaptive_adj --adaptive_adj_dim 32
#   --loss L1
# Variants:
#   A1 = R - Prior1 (drop --use_distance_anchor --use_coord_embed)
#   A2 = R - Prior2 (drop --use_laplacian_smooth; keep graph_prop because
#                    Mode 2 depends on it; A2 isolates the loss-side penalty)
#   A3 = R - Mode1  (drop --use_dense_attn)
#   A4 = R - Mode2  (drop --use_adaptive_adj; keep graph_prop for Prior 2)
#
# Channels (5): pm25, ghi, tmax, prcp, mob-outdoor.
# Excluded (large-N mob-essential/indoor/food): R requires --use_dense_attn,
# which OOMs on N>6286 per Tab 1's failed indoor_outdoor_pattern run.
#
# 5 channels x 5 variants = 25 jobs.

set -euo pipefail

CHANNELS=(
  "CalGeo-AirQuality-pm25"
  "CalGeo-Solar-ghi"
  "CalGeo-Weather-tmax"
  "CalGeo-Weather-prcp"
  "Mobility-CA-outdoor"
)

# Variant -> flag string mapping. R has all components; Aj drops one.
declare -A VARIANT_FLAGS
VARIANT_FLAGS[R]="--use_distance_anchor --distance_alpha 1.0 --use_coord_embed --use_laplacian_smooth --laplacian_lambda 0.05 --use_graph_prop --use_dense_attn --use_adaptive_adj --adaptive_adj_dim 32"
VARIANT_FLAGS[A1]="--use_laplacian_smooth --laplacian_lambda 0.05 --use_graph_prop --use_dense_attn --use_adaptive_adj --adaptive_adj_dim 32"
VARIANT_FLAGS[A2]="--use_distance_anchor --distance_alpha 1.0 --use_coord_embed --use_graph_prop --use_dense_attn --use_adaptive_adj --adaptive_adj_dim 32"
VARIANT_FLAGS[A3]="--use_distance_anchor --distance_alpha 1.0 --use_coord_embed --use_laplacian_smooth --laplacian_lambda 0.05 --use_graph_prop --use_adaptive_adj --adaptive_adj_dim 32"
VARIANT_FLAGS[A4]="--use_distance_anchor --distance_alpha 1.0 --use_coord_embed --use_laplacian_smooth --laplacian_lambda 0.05 --use_graph_prop --use_dense_attn"

VARIANTS=(R A1 A2 A3 A4)
COMMON=(--report_raw_metrics --train_epochs 100 --patience 15 --batch_size 64 --loss L1)

MANIFEST="${IJGIS_ROOT:-../ijgis}/v4/ablation_sweep_manifest.tsv"
mkdir -p "$(dirname "$MANIFEST")"
echo -e "channel\tvariant\tjob_id\tflags\tsubmitted_at" > "$MANIFEST"

submit() {
  local chan="$1"; local var="$2"; local flags="$3"
  local short_chan="${chan##*-}"  # last hyphen-segment, e.g. pm25, ghi, outdoor
  local des="ablation_${short_chan}_${var}"
  local out
  out=$(bash scripts/submit_ray_spatiallca.sh --model RegionFormer "$chan" \
        --des "$des" \
        "${COMMON[@]}" $flags 2>&1)
  local job
  job=$(echo "$out" | grep -oE 'raysubmit_[A-Za-z0-9]+' | head -1)
  echo "  [$short_chan/$var] -> $job"
  printf "%s\t%s\t%s\t%s\t%s\n" "$chan" "$var" "$job" "$flags" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
}

echo "Submitting 5 channels x 5 variants = 25 ablation jobs..."
for chan in "${CHANNELS[@]}"; do
  for var in "${VARIANTS[@]}"; do
    submit "$chan" "$var" "${VARIANT_FLAGS[$var]}"
  done
done

echo
echo "Submitted 25 ablation jobs. Manifest: $MANIFEST"
