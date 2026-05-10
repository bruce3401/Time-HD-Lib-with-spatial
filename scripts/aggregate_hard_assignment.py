#!/usr/bin/env python3
"""Aggregate §6.2 hard-assignment ablation results.

For each of 5 channels, reads:
  - the soft-R cell from ijgis/v4/ablation_sweep_metrics.json (already on disk)
  - the hard-W cell from wandb (matched by config.des = ablation_<chan>_hardW)

Produces ijgis/v4/soft_vs_hard.json with per-channel {soft_mae, hard_mae,
soft_rmse, hard_rmse, delta_pct} cells, plus a print of the LaTeX rows ready
to paste into the tab:soft-vs-hard table in main.tex.
"""

import json
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
IJGIS = ROOT.parent / "ijgis"
SOFT_SRC = IJGIS / "v4" / "ablation_sweep_metrics.json"
DST = IJGIS / "v4" / "soft_vs_hard.json"

CHANNELS = [
    ("CalGeo-AirQuality-pm25", "pm25",        r"\textsc{Pm2.5}"),
    ("CalGeo-Solar-ghi",        "ghi",         r"\textsc{Ghi}"),
    ("CalGeo-Weather-tmax",     "tmax",        r"\textsc{Tmax}"),
    ("CalGeo-Weather-prcp",     "prcp",        r"\textsc{Prcp}"),
    ("Mobility-CA-outdoor",     "mob-outdoor", r"\textsc{Mob-Outdoor}"),
]
SHORT2FULL = {short: full for full, short, _ in CHANNELS}


def fetch_hard():
    api = wandb.Api()
    by_des = {}
    for r in api.runs(f"{api.default_entity}/spatialscale"):
        des = r.config.get("des", "")
        if des.startswith("ablation_") and des.endswith("_hardW"):
            if des not in by_des or r.created_at > by_des[des].created_at:
                by_des[des] = r
    out = {}
    for full, short, _ in CHANNELS:
        # short channel for des — last hyphen segment, matching sweep_hard_assignment.sh
        des_key = "ablation_" + full.split("-")[-1] + "_hardW"
        run = by_des.get(des_key)
        if run is None:
            out[short] = None
        else:
            out[short] = {
                "mae":  run.summary.get("final_raw_mae"),
                "rmse": run.summary.get("final_raw_rmse"),
                "wandb": run.url,
                "status": run.state,
            }
    return out


def fmt(x, scale=1.0):
    if x is None:
        return r"\textsc{tbd}"
    v = x * scale
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v) >= 100:  return f"{v:.1f}"
    if abs(v) >= 10:   return f"{v:.2f}"
    if abs(v) >= 1:    return f"{v:.3f}"
    return f"{v:.4f}"


def main():
    soft = json.loads(SOFT_SRC.read_text())  # keyed by short chan -> {R, A1, ...}
    hard = fetch_hard()
    out = {}
    print("\nLatex rows for table tab:soft-vs-hard:\n")
    for full, short, label in CHANNELS:
        s = soft.get(short, {}).get("R") or {}
        h = hard.get(short) or {}
        s_mae, s_rmse = s.get("mae"), s.get("rmse")
        h_mae, h_rmse = h.get("mae"), h.get("rmse")
        delta = None
        if s_mae and h_mae:
            delta = (h_mae - s_mae) / s_mae * 100
        out[short] = {
            "label": label,
            "soft":  {"mae": s_mae, "rmse": s_rmse},
            "hard":  {"mae": h_mae, "rmse": h_rmse, "wandb": h.get("wandb")},
            "delta_pct_mae": delta,
        }
        # Ozone display would scale x1000 (ppb), but ozone isn't in our 5 channels.
        scale = 1.0
        s_str = fmt(s_mae, scale)
        h_str = fmt(h_mae, scale)
        d_str = f"+{delta:.1f}\\%" if delta is not None else "—"
        print(f"  {label:24s} & {s_str} & {h_str} & {d_str} \\\\")
    DST.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {DST}")


if __name__ == "__main__":
    main()
