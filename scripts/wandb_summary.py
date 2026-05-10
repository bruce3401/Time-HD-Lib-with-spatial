"""Pull a compact summary table from wandb project `spatialscale`.

Usage:
  python scripts/wandb_summary.py [--filter <substring>] [--state finished]

Reports per run: name, state, key z-/raw-metrics from summary, runtime, url.
Requires `wandb login` to have been done locally.
"""
from __future__ import annotations
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "your-wandb-entity"))
    ap.add_argument("--project", default="spatialscale")
    ap.add_argument("--filter", default=None,
                    help="substring filter on run name (e.g. 'gba_raw')")
    ap.add_argument("--state", default=None,
                    help="finished | running | crashed | killed")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. `pip install wandb` first.", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", per_page=200)
    rows = []
    for r in runs:
        if args.filter and args.filter not in r.name:
            continue
        if args.state and r.state != args.state:
            continue
        s = r.summary
        rows.append({
            "name": r.name,
            "state": r.state,
            "z_mae": s.get("final_test_mae", float("nan")),
            "z_mse": s.get("final_test_mse", float("nan")),
            "raw_mae": s.get("final_raw_mae", float("nan")),
            "raw_rmse": s.get("final_raw_rmse", float("nan")),
            "raw_mape": s.get("final_raw_mape", float("nan")),
            "url": r.url,
        })
        if len(rows) >= args.limit:
            break

    if not rows:
        print("no matching runs")
        return

    rows.sort(key=lambda r: (r["raw_mae"] if r["raw_mae"] == r["raw_mae"] else 1e9))

    print(f"{'name':<55} {'state':<10} {'raw_mae':>9} {'raw_rmse':>9} {'raw_mape':>9} {'z_mae':>7}")
    for r in rows:
        try:
            mape_pct = r['raw_mape'] * 100
        except (TypeError, ValueError):
            mape_pct = float('nan')
        print(f"{r['name'][:55]:<55} {r['state']:<10} {r['raw_mae']:>9.3f} {r['raw_rmse']:>9.3f} {mape_pct:>8.2f}% {r['z_mae']:>7.4f}")


if __name__ == "__main__":
    main()
