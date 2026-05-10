#!/usr/bin/env python3
"""Aggregate §6 ablation sweep results from wandb.

Reads `ijgis/v4/ablation_sweep_manifest.tsv` (channel, variant, ray_job_id,
flags, submitted_at) and pulls each run's `final_raw_mae`, `final_raw_rmse`
(or computes RMSE from `final_test_mse` after inverse-transform), and
`final_val_raw_mae` from the `spatialscale` wandb project.

Writes:
  ijgis/v4/ablation_sweep_results.json — flat list with metrics per cell
  ijgis/v4/ablation_sweep_metrics.json — pivoted (channel x variant) matrix
                                          ready for table/figure generation

Usage:  python scripts/aggregate_ablation_sweep.py
"""

import json
import math
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed in this env.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
IJGIS = ROOT.parent / "ijgis"
MANIFEST = IJGIS / "v4" / "ablation_sweep_manifest.tsv"
RESULTS_JSON = IJGIS / "v4" / "ablation_sweep_results.json"
METRICS_JSON = IJGIS / "v4" / "ablation_sweep_metrics.json"

CHANNELS = ["CalGeo-AirQuality-pm25", "CalGeo-Solar-ghi",
            "CalGeo-Weather-tmax", "CalGeo-Weather-prcp",
            "Mobility-CA-outdoor"]
VARIANTS = ["R", "A1", "A2", "A3", "A4"]
SUMMARY_KEYS = ["final_raw_mae", "final_raw_rmse", "final_val_raw_mae",
                "final_test_mse", "final_test_mae"]


def read_manifest():
    rows = []
    with open(MANIFEST) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2]:
                rows.append({"channel": parts[0], "variant": parts[1],
                             "ray_id": parts[2],
                             "flags": parts[3] if len(parts) > 3 else ""})
    return rows


def short_chan_for_des(c):
    """Match the suffix used in --des in sweep_ablation.sh."""
    return c.split("-")[-1]  # pm25, ghi, tmax, prcp, outdoor


def fetch_wandb_metrics(rows):
    api = wandb.Api()
    proj = f"{api.default_entity}/spatialscale"
    # Index runs by `des` field (which encodes channel+variant via our --des tag)
    by_des = {}
    for r in api.runs(proj):
        des = r.config.get("des", "")
        if des.startswith("ablation_"):
            # keep the most recently created run for each des (in case of retries)
            if des not in by_des or r.created_at > by_des[des].created_at:
                by_des[des] = r
    out = []
    for row in rows:
        des_key = f"ablation_{short_chan_for_des(row['channel'])}_{row['variant']}"
        run = by_des.get(des_key)
        rec = dict(row)
        rec["des_key"] = des_key
        if run is None:
            for k in SUMMARY_KEYS:
                rec[k] = None
            rec["wandb_url"] = None
            rec["status"] = "NOT_FOUND"
        else:
            for k in SUMMARY_KEYS:
                rec[k] = run.summary.get(k)
            rec["wandb_url"] = run.url
            rec["status"] = run.state
        out.append(rec)
    return out


def short_chan(c):
    return {
        "CalGeo-AirQuality-pm25": "pm25",
        "CalGeo-Solar-ghi": "ghi",
        "CalGeo-Weather-tmax": "tmax",
        "CalGeo-Weather-prcp": "prcp",
        "Mobility-CA-outdoor": "mob-outdoor",
    }[c]


def main():
    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST}")
        sys.exit(1)
    rows = read_manifest()
    print(f"Manifest has {len(rows)} entries.")
    rows = fetch_wandb_metrics(rows)
    n_ok = sum(1 for r in rows if r.get("final_raw_mae") is not None)
    print(f"final_raw_mae present: {n_ok}/{len(rows)}")

    RESULTS_JSON.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {RESULTS_JSON}")

    pivot = {short_chan(c): {} for c in CHANNELS}
    for r in rows:
        ch = short_chan(r["channel"])
        v = r["variant"]
        mae = r.get("final_raw_mae")
        rmse = r.get("final_raw_rmse")
        if rmse is None and r.get("final_test_mse") is not None:
            rmse = math.sqrt(r["final_test_mse"])
        pivot[ch][v] = {"mae": mae, "rmse": rmse,
                        "val_mae": r.get("final_val_raw_mae"),
                        "wandb": r.get("wandb_url"),
                        "status": r.get("status")}
    METRICS_JSON.write_text(json.dumps(pivot, indent=2))
    print(f"Wrote {METRICS_JSON}")

    # Quick console summary
    print("\nChannel × Variant raw MAE:")
    print(f"{'channel':14}{' '.join(f'{v:>10}' for v in VARIANTS)}")
    for c in CHANNELS:
        ch = short_chan(c)
        cells = []
        for v in VARIANTS:
            x = pivot[ch].get(v, {}).get("mae")
            cells.append(f"{x:>10.4g}" if isinstance(x, (int, float)) else f"{'-':>10}")
        print(f"{ch:14}" + "".join(cells))


if __name__ == "__main__":
    main()
