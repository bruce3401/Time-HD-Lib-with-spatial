#!/usr/bin/env python3
"""Aggregate baseline sweep results from wandb into a (model x channel) z-MSE table.

Reads the manifest at ../ijgis/v4/baseline_sweep_manifest.tsv (model, channel,
ray_job_id, submitted_at), pulls each run's `final_test_mse`, `final_test_mae`,
`final_raw_mae`, and `final_val_raw_mae` from the `spatialscale` wandb project
by matching the ray submission ID stored on each run, then writes:

  ../ijgis/v4/baseline_sweep_results.json   — flat list, all 4 metrics per cell
  ../ijgis/v4/baseline_sweep_zmse.tsv       — model x channel z-MSE matrix
  ../ijgis/v4/baseline_sweep_rawmae.tsv     — model x channel raw-MAE matrix

Also pulls the per-channel RegionFormer winner z-MSE from
ijgis/v4/final_verdict.json so the matrix has 8 rows (RF + 7 baselines).

Usage:  python scripts/aggregate_baseline_sweep.py
"""

import json
import os
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed in this env. Run inside the cluster venv.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
IJGIS = ROOT.parent / "ijgis"
MANIFEST = IJGIS / "v4" / "baseline_sweep_manifest.tsv"
RESULTS_JSON = IJGIS / "v4" / "baseline_sweep_results.json"
ZMSE_TSV = IJGIS / "v4" / "baseline_sweep_zmse.tsv"
RAWMAE_TSV = IJGIS / "v4" / "baseline_sweep_rawmae.tsv"
RF_VERDICT = IJGIS / "v4" / "final_verdict.json"

CHANNELS = [
    "CalGeo-AirQuality-pm25",
    "CalGeo-AirQuality-ozone",
    "CalGeo-Solar-ghi",
    "CalGeo-Weather-tmax",
    "CalGeo-Weather-tmin",
    "CalGeo-Weather-prcp",
    "Mobility-CA-outdoor",
    "Mobility-CA-essential",
    "Mobility-CA-indoor",
    "Mobility-CA-food",
]
BASELINES = ["DLinear", "iTransformer", "PatchTST", "TimeMixer",
             "ModernTCN", "CycleNet", "TSMixer"]
METRICS = ["final_test_mse", "final_test_mae", "final_raw_mae",
           "final_val_raw_mae"]


def read_manifest():
    rows = []
    with open(MANIFEST) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[2]:
                rows.append({"model": parts[0], "channel": parts[1],
                             "ray_id": parts[2]})
    return rows


def fetch_wandb_metrics(rows):
    api = wandb.Api()
    project = "spatialscale"
    runs = list(api.runs(f"{api.default_entity}/{project}",
                         filters={"$or": [{"config.ray_job_id": r["ray_id"]}
                                          for r in rows[:1]]}))
    print(f"Pulling {len(rows)} runs from wandb project '{project}'...")
    by_id = {}
    for r in api.runs(f"{api.default_entity}/{project}"):
        rid = r.config.get("ray_job_id") or r.summary.get("ray_job_id")
        if rid:
            by_id[rid] = r
    out = []
    for row in rows:
        run = by_id.get(row["ray_id"])
        rec = dict(row)
        if run is None:
            rec["wandb_url"] = None
            for m in METRICS:
                rec[m] = None
            rec["status"] = "NOT_FOUND"
        else:
            rec["wandb_url"] = run.url
            for m in METRICS:
                rec[m] = run.summary.get(m)
            rec["status"] = run.state
        out.append(rec)
    return out


def fetch_rf_zmse_via_wandb(rf_verdict):
    """For each channel, RF winner is referenced by sid in final_verdict.json
    — pull its final_test_mse from wandb."""
    api = wandb.Api()
    proj_path = f"{api.default_entity}/spatialscale"
    sid_to_chan = {}
    for chan_key, info in rf_verdict.items():
        sid = info.get("rf_sid")
        if sid:
            sid_to_chan[sid] = chan_key
    out = {}
    for r in api.runs(proj_path):
        rid = r.config.get("ray_job_id") or r.summary.get("ray_job_id")
        if rid in sid_to_chan:
            chan = sid_to_chan[rid]
            out[chan] = {m: r.summary.get(m) for m in METRICS}
            out[chan]["wandb_url"] = r.url
    return out


def build_matrix(rows, rf_zmse, metric):
    # Map channel string used in manifest -> verdict key
    chan2vkey = {
        "CalGeo-AirQuality-pm25": "pm25",
        "CalGeo-AirQuality-ozone": "ozone",
        "CalGeo-Solar-ghi": "ghi",
        "CalGeo-Weather-tmax": "tmax",
        "CalGeo-Weather-tmin": "tmin",
        "CalGeo-Weather-prcp": "prcp",
        "Mobility-CA-outdoor": "mob-outdoor",
        "Mobility-CA-essential": "mob-essential",
        "Mobility-CA-indoor": "mob-indoor",
        "Mobility-CA-food": "mob-food",
    }
    by_cell = {(r["model"], r["channel"]): r.get(metric) for r in rows}
    short_chans = [chan2vkey[c] for c in CHANNELS]
    lines = ["model\t" + "\t".join(short_chans)]
    # RegionFormer row first
    rf_row = ["RegionFormer"]
    for c in CHANNELS:
        v = rf_zmse.get(chan2vkey[c], {}).get(metric)
        rf_row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
    lines.append("\t".join(rf_row))
    for m in BASELINES:
        row = [m]
        for c in CHANNELS:
            v = by_cell.get((m, c))
            row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
        lines.append("\t".join(row))
    return "\n".join(lines)


def main():
    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST}")
        sys.exit(1)
    rows = read_manifest()
    print(f"Manifest has {len(rows)} entries.")
    rows_with_metrics = fetch_wandb_metrics(rows)
    rf_verdict = json.load(open(RF_VERDICT))
    rf_zmse = fetch_rf_zmse_via_wandb(rf_verdict)
    print(f"Fetched RF metrics for {len(rf_zmse)} channels.")

    RESULTS_JSON.write_text(json.dumps(
        {"baseline_runs": rows_with_metrics,
         "rf_winner_metrics": rf_zmse}, indent=2))
    print(f"Wrote {RESULTS_JSON}")

    ZMSE_TSV.write_text(build_matrix(rows_with_metrics, rf_zmse,
                                     "final_test_mse"))
    RAWMAE_TSV.write_text(build_matrix(rows_with_metrics, rf_zmse,
                                       "final_raw_mae"))
    print(f"Wrote {ZMSE_TSV}\nWrote {RAWMAE_TSV}")

    n_ok = sum(1 for r in rows_with_metrics
               if r.get("final_test_mse") is not None)
    print(f"\nz-MSE values present: {n_ok}/{len(rows_with_metrics)}")


if __name__ == "__main__":
    main()
