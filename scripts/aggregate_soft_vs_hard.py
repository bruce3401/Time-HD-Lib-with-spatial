#!/usr/bin/env python3
"""§6.2 Soft-vs-hard ablation aggregator (extended 8-channel version).

Reads ijgis/v4/soft_vs_hard_manifest.tsv and pulls each cell's final_raw_mae
+ final_raw_rmse from wandb (matched by config.des = ablation_<chan>_R2_<var>).

Writes ijgis/v4/soft_vs_hard.json and prints the LaTeX rows ready to paste
into the tab:soft-vs-hard table in main.tex.
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
MANIFEST = IJGIS / "v4" / "soft_vs_hard_manifest.tsv"
DST = IJGIS / "v4" / "soft_vs_hard.json"

CHAN_ORDER = [
    ("CalGeo-AirQuality-pm25",  "pm25",          r"\textsc{Pm2.5}",        1.0),
    ("CalGeo-Solar-ghi",         "ghi",           r"\textsc{Ghi}",          1.0),
    ("CalGeo-Weather-tmax",      "tmax",          r"\textsc{Tmax}",         1.0),
    ("CalGeo-Weather-prcp",      "prcp",          r"\textsc{Prcp}",         1.0),
    ("Mobility-CA-outdoor",      "outdoor",       r"\textsc{Mob-Outdoor}",  1.0),
    ("Mobility-CA-essential",    "essential",     r"\textsc{Mob-Essential}",1.0),
    ("Mobility-CA-indoor",       "indoor",        r"\textsc{Mob-Indoor}",   1.0),
    ("Mobility-CA-food",         "food",          r"\textsc{Mob-Food}",     1.0),
]


def fetch(rev_tag="R2"):
    """rev_tag: which R revision to fetch — R2 (topk K=8) or R3 (r=32+gumbel)."""
    api = wandb.Api()
    needle = f"_{rev_tag}_"
    by_des = {}
    for r in api.runs(f"{api.default_entity}/spatialscale"):
        des = r.config.get("des", "")
        if des.startswith("ablation_") and needle in des:
            if des not in by_des or r.created_at > by_des[des].created_at:
                by_des[des] = r
    return by_des


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
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--rev", choices=["R2", "R3"], default="R2",
                   help="R2 = topk K=8 (Stage 1); R3 = r=32 + gumbel (Stage 2).")
    args = p.parse_args()
    rev = args.rev
    by_des = fetch(rev)
    out = {}
    print(f"\nLatex rows for table tab:soft-vs-hard ({rev}):\n")
    for full, short, label, scale in CHAN_ORDER:
        soft_run = by_des.get(f"ablation_{short}_{rev}_softW")
        hard_run = by_des.get(f"ablation_{short}_{rev}_hardW")
        s_mae = soft_run.summary.get("final_raw_mae") if soft_run else None
        h_mae = hard_run.summary.get("final_raw_mae") if hard_run else None
        delta = (h_mae - s_mae) / s_mae * 100 if (s_mae and h_mae) else None
        out[short] = {
            "label": label,
            "soft":  {"mae": s_mae, "wandb": soft_run.url if soft_run else None,
                      "status": soft_run.state if soft_run else "MISSING"},
            "hard":  {"mae": h_mae, "wandb": hard_run.url if hard_run else None,
                      "status": hard_run.state if hard_run else "MISSING"},
            "delta_pct_mae": delta,
        }
        s_str = fmt(s_mae, scale)
        h_str = fmt(h_mae, scale)
        if delta is None:
            d_str = "---"
        else:
            d_str = f"${delta:+.1f}\\%$"
        # Bold the lower of the two
        if s_mae and h_mae:
            if s_mae < h_mae:
                s_str = r"\textbf{" + s_str + "}"
            elif h_mae < s_mae:
                h_str = r"\textbf{" + h_str + "}"
        print(f"  {label:24s} & {s_str:24s} & {h_str:24s} & {d_str} \\\\")
    out_path = DST.with_name(f"soft_vs_hard_{rev}.json")
    out_path.write_text(json.dumps(out, indent=2))
    wins = sum(1 for v in out.values()
               if v["soft"]["mae"] is not None and v["hard"]["mae"] is not None
               and v["soft"]["mae"] < v["hard"]["mae"])
    total = sum(1 for v in out.values()
                if v["soft"]["mae"] is not None and v["hard"]["mae"] is not None)
    print(f"\nSoft wins {wins}/{total} channels ({rev})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
