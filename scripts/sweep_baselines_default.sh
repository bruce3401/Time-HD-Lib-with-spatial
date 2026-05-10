#!/usr/bin/env bash
# Full baseline sweep: 7 models x 10 CalGeo channels at default config,
# matched training budget (100 epochs + patience 15 early stopping).
#
# Architecture/optimizer/lr come from configs/<MODEL>.yaml per-channel section.
# Only training-budget knobs are matched to RegionFormer's protocol.
# batch_size scaled per channel-N to fit GPU comfortably.
#
# Usage: bash scripts/sweep_baselines_default.sh [--dry-run]

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; fi

MODELS=(DLinear iTransformer PatchTST TimeMixer ModernTCN CycleNet TSMixer)

CHANNELS=(
  CalGeo-AirQuality-pm25
  CalGeo-AirQuality-ozone
  CalGeo-Solar-ghi
  CalGeo-Weather-tmax
  CalGeo-Weather-tmin
  CalGeo-Weather-prcp
  Mobility-CA-outdoor
  Mobility-CA-essential
  Mobility-CA-indoor
  Mobility-CA-food
)

# Per-channel batch_size (memory-aware; doesn't change model capacity)
declare -A BATCH=(
  [CalGeo-AirQuality-pm25]=64
  [CalGeo-AirQuality-ozone]=64
  [CalGeo-Solar-ghi]=64
  [CalGeo-Weather-tmax]=64
  [CalGeo-Weather-tmin]=64
  [CalGeo-Weather-prcp]=64
  [Mobility-CA-outdoor]=64
  [Mobility-CA-essential]=16
  [Mobility-CA-indoor]=16
  [Mobility-CA-food]=8
)

MANIFEST="${MANIFEST:-../ijgis/v4/baseline_sweep_manifest.tsv}"
mkdir -p "$(dirname "$MANIFEST")"
if [ ! -f "$MANIFEST" ]; then
  echo -e "model\tchannel\tjob_id\tsubmitted_at" > "$MANIFEST"
fi

# Cells already covered (model:channel pairs) — skip on relaunch
SKIP="DLinear:CalGeo-AirQuality-pm25"

n=0
for model in "${MODELS[@]}"; do
  for chan in "${CHANNELS[@]}"; do
    n=$((n+1))
    if [[ ":$SKIP:" == *":${model}:${chan}:"* ]]; then
      echo "[$n/70] SKIP $model on $chan (already done)"
      continue
    fi
    bs=${BATCH[$chan]}
    common_args=(
      --train_epochs 100
      --patience 15
      --batch_size "$bs"
      --report_raw_metrics
    )

    if [ "$DRY_RUN" = "1" ]; then
      echo "[$n] bash scripts/submit_ray_spatiallca.sh --model $model $chan ${common_args[*]}"
      continue
    fi

    echo "[$n/70] launching $model on $chan (bs=$bs)..."
    out=$(bash scripts/submit_ray_spatiallca.sh --model "$model" "$chan" "${common_args[@]}" 2>&1)
    job_id=$(echo "$out" | grep -oE 'raysubmit_[A-Za-z0-9]+' | head -1)
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo -e "${model}\t${chan}\t${job_id}\t${ts}" >> "$MANIFEST"
    echo "  -> ${job_id}"
    sleep 1
  done
done

echo ""
echo "Submitted $n jobs. Manifest: $MANIFEST"
echo "Watch: ray job list --address ${RAY_ADDR}"
