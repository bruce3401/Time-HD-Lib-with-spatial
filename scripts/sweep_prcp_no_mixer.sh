#!/usr/bin/env bash
# Prcp-focused sweep — find the best RegionFormer recipe WITHOUT the mixer/PatchTST
# branch capabilities. Goal: see if RF's spatial priors + capabilities alone can
# beat TSMixer's default-config 3.457 raw-MAE on prcp under s=2021.
#
# Common: 100 epochs + patience 15, default batch, single seed s=2021.
# All 14 variants captured per-row in $MANIFEST below.

set -euo pipefail
DATA=CalGeo-Weather-prcp
COMMON=(--report_raw_metrics --train_epochs 100 --patience 15 --batch_size 64)
MANIFEST="${IJGIS_ROOT:-../ijgis}/v4/prcp_no_mixer_sweep_manifest.tsv"
mkdir -p "$(dirname "$MANIFEST")"
echo -e "tag\tjob_id\tflags\tsubmitted_at" > "$MANIFEST"

submit() {
  local tag="$1"; shift
  local extra=("$@")
  local out
  out=$(bash scripts/submit_ray_spatiallca.sh --model RegionFormer "$DATA" \
        --des "prcp_nomix_${tag}" \
        "${COMMON[@]}" "${extra[@]}" 2>&1)
  local job
  job=$(echo "$out" | grep -oE 'raysubmit_[A-Za-z0-9]+' | head -1)
  echo "  [$tag] -> $job"
  printf "%s\t%s\t%s\t%s\n" "$tag" "$job" "${extra[*]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
}

submit "01_rf_default"           # bare RF
submit "02_dense_L1"              --use_dense_attn --loss L1
submit "03_dense_L1_lion"         --use_dense_attn --loss L1 --optimizer lion
submit "04_distance_anchor"       --use_distance_anchor --distance_alpha 1.0
submit "05_laplacian_lam05"       --use_laplacian_smooth --laplacian_lambda 0.05
submit "06_priors_both"           --use_distance_anchor --distance_alpha 1.0 --use_laplacian_smooth --laplacian_lambda 0.05
submit "07_daily_full"            --use_dense_attn --loss L1 --optimizer lion --use_distance_anchor --distance_alpha 1.0 --use_laplacian_smooth --laplacian_lambda 0.05
submit "08_tmin_recipe"           --use_coord_embed --use_graph_prop --use_dense_attn --loss L1
submit "09_mob_style"             --use_graph_prop --use_adaptive_adj --adaptive_adj_dim 32 --loss L1
submit "10_sl28_dense"            --seq_len 28 --use_dense_attn --loss L1
submit "11_sl42_dense_priors"     --seq_len 42 --use_dense_attn --loss L1 --use_distance_anchor --distance_alpha 1.0 --use_laplacian_smooth --laplacian_lambda 0.05
submit "12_tmax_recipe"           --seq_len 28 --use_dense_attn --loss L1 --use_laplacian_smooth --laplacian_lambda 0.10
submit "13_d_model_512"           --d_model 512 --use_dense_attn --loss L1
submit "14_huber"                 --use_dense_attn --loss huber

echo
echo "Submitted 14 prcp jobs. Manifest: $MANIFEST"
