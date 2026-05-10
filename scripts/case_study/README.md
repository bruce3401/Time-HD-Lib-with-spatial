# Case Study Pipeline — IJGIS §7

Two case studies share one architectural payload: RegionFormer's soft assignment $\bm{W}$.

- **Case 1** (§7.1): Latent regions rediscover California's geographic structure.
- **Case 2** (§7.2): Same regions reorganize under heat events; per-tract surprise tracks CalEnviroScreen vulnerability.

See `ijgis/v4/case_study/PLAN.md` for narrative + status. This README is the operational runbook.

## Pipeline

```
┌──── cluster ────┐                  ┌──── local ────┐
01_extract_W.py    →   W_argmax_<ch>.npz   →   02_align_to_ref.py   →   case1_alignment_*.json
03_extract_heat_residuals.py  →   heat_residuals_<ch>.npz  →   04_heat_correlation.py   →   case2_correlation_*.json
                                                                      ↓
                                           05_make_maps.py  (TODO)   →  manuscripts/Figures/case{1,2}_*.png
```

## Step-by-step

### 0. External data (run **once**, on local machine)

Place these under `ijgis/v4/case_study/data/`:

| File | Source | Purpose |
|---|---|---|
| `tract_to_epa_l3.csv` | EPA L3 ecoregions shapefile + spatial join to TIGER tract centroids | Case 1 reference geography |
| `tract_to_county.csv` | TIGER 2020 (county FIPS already in `<channel>_meta.csv`) | Case 1 reference geography |
| `tract_to_climate_zone.csv` | CARB air basins or CIMIS ETo zones | Case 1 reference geography |
| `ces4_tract.csv` | CalEnviroScreen 4.0 OEHHA download | Case 2 vulnerability validation |
| `coastal_distance.csv` (optional) | distance to PCH from each tract centroid | Case 2 inland/coastal split |

Schema: `GEOID,ref_region` for each `*_to_*` (ref) file; `GEOID,col1,col2,...` for `ces4_tract.csv`.

### 1. Extract W (cluster, GPU)

```bash
cd Time-HD-Lib-with-spatial
for ch in Mobility-CA-outdoor CalGeo-AirQuality-pm25 CalGeo-Solar-ghi; do
  python scripts/case_study/01_extract_W.py \
    --ckpt /data/checkpoints-timehd/<RF_ckpt_for_${ch}>.pth \
    --data "$ch" \
    --root_path /data/timehd/${ch%%-*}/ \
    --out ${IJGIS_ROOT:-../ijgis}/v4/case_study/outputs/W_argmax_${ch}.npz
done
```

### 2. Align W to reference geographies (local)

```bash
for ch in Mobility-CA-outdoor CalGeo-AirQuality-pm25 CalGeo-Solar-ghi; do
  for ref in epa_l3 county climate_zone; do
    python scripts/case_study/02_align_to_ref.py \
      --W ijgis/v4/case_study/outputs/W_argmax_${ch}.npz \
      --ref ijgis/v4/case_study/data/tract_to_${ref}.csv \
      --label ${ref} \
      --out ijgis/v4/case_study/outputs/
  done
done
```

### 3. Extract heat-vs-non-heat residuals (cluster, GPU)

```bash
for ch in Mobility-CA-outdoor Mobility-CA-essential Mobility-CA-indoor Mobility-CA-food; do
  python scripts/case_study/03_extract_heat_residuals.py \
    --ckpt /data/checkpoints-timehd/<RF_ckpt_for_${ch}>.pth \
    --data "$ch" \
    --root_path /data/timehd/Mobility_CA/ \
    --heat_weeks_csv /data/timehd/NOAA/heat_weeks_2018_2020.csv \
    --out ${IJGIS_ROOT:-../ijgis}/v4/case_study/outputs/heat_residuals_${ch}.npz
done
```

### 4. Heat-event correlation analysis (local)

```bash
for ch in Mobility-CA-outdoor Mobility-CA-essential Mobility-CA-indoor Mobility-CA-food; do
  python scripts/case_study/04_heat_correlation.py \
    --residuals ijgis/v4/case_study/outputs/heat_residuals_${ch}.npz \
    --ces4 ijgis/v4/case_study/data/ces4_tract.csv \
    --coastal_distance_csv ijgis/v4/case_study/data/coastal_distance.csv \
    --out ijgis/v4/case_study/outputs/
done
```

### 5. Make publication maps (TODO)

`05_make_maps.py` will produce the §7.1 / §7.2 figure PNGs at 300 DPI from
the JSON + npz outputs above.

## Status

- ✅ Scripts 01–04 written
- ⏳ 05 (figures) pending
- ⏳ External data download URLs being investigated
- ⏳ Cluster runs (01, 03) pending checkpoint paths
