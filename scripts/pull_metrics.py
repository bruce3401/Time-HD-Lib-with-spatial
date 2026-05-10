"""Pull final metrics from a list of Ray job IDs.

Usage:
  python scripts/pull_metrics.py <jid1> [jid2 ...]
  python scripts/pull_metrics.py --tags  jid1:tag1 jid2:tag2 ...

Reports per-job: status | val-raw-MAE | raw-MAE / RMSE / MAPE | wandb URL.

val-raw-MAE is the SELECTION metric (post 2026-05-08 audit, see findings.md).
raw-MAE is the REPORTING metric for the val-selected configuration.
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys

RAY_ADDR = os.environ.get("RAY_ADDR", "http://localhost:8265")

PAT_TEST = re.compile(r"Test MSE:\s*([0-9.eE+-]+),\s*Test MAE:\s*([0-9.eE+-]+)")
PAT_RAW = re.compile(r"\[raw\] MAE=([0-9.]+)\s+RMSE=([0-9.]+)\s+MAPE=([0-9.]+)%")
PAT_VAL_RAW = re.compile(r"\[val-raw\] MAE=([0-9.]+)\s+RMSE=([0-9.]+)\s+MAPE=([0-9.]+)%")
PAT_WANDB_URL = re.compile(r"https://wandb\.ai/[A-Za-z0-9_/.-]+/runs/[a-z0-9]+")


def fetch_log(jid: str) -> str:
    return subprocess.run(
        ["ray", "job", "logs", jid, "--address", RAY_ADDR],
        check=False, capture_output=True, text=True
    ).stdout + subprocess.run(
        ["ray", "job", "logs", jid, "--address", RAY_ADDR],
        check=False, capture_output=True, text=True
    ).stderr


def fetch_status(jid: str) -> str:
    return subprocess.run(
        ["curl", "-s", f"{RAY_ADDR}/api/jobs/{jid}"],
        check=False, capture_output=True, text=True
    ).stdout


def parse_metrics(log: str) -> dict:
    out = {}
    m = PAT_TEST.search(log)
    if m:
        out["z_mse"] = float(m.group(1))
        out["z_mae"] = float(m.group(2))
    m = PAT_RAW.search(log)
    if m:
        out["raw_mae"] = float(m.group(1))
        out["raw_rmse"] = float(m.group(2))
        out["raw_mape"] = float(m.group(3)) / 100  # convert percent
    m = PAT_VAL_RAW.search(log)
    if m:
        out["val_raw_mae"] = float(m.group(1))
        out["val_raw_rmse"] = float(m.group(2))
        out["val_raw_mape"] = float(m.group(3)) / 100
    m = PAT_WANDB_URL.search(log)
    if m:
        out["wandb_url"] = m.group(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", help="Ray job IDs (or jid:tag pairs)")
    args = ap.parse_args()

    print(f"{'tag':<28} {'status':<11} {'val-MAE':>9} {'raw-MAE':>9} {'raw-RMSE':>9} {'raw-MAPE':>9} {'z-MAE':>8} url")
    for spec in args.jobs:
        if ":" in spec:
            jid, tag = spec.split(":", 1)
        else:
            jid, tag = spec, spec[-12:]
        import json
        try:
            status = json.loads(fetch_status(jid)).get("status", "?")
        except Exception:
            status = "?"
        log = fetch_log(jid)
        m = parse_metrics(log)
        z_mae = m.get("z_mae", float("nan"))
        rm = m.get("raw_mae", float("nan"))
        rr = m.get("raw_rmse", float("nan"))
        rp = m.get("raw_mape", float("nan"))
        vrm = m.get("val_raw_mae", float("nan"))
        url = m.get("wandb_url", "")
        print(f"{tag:<28} {status:<11} {vrm:>9.3f} {rm:>9.3f} {rr:>9.3f} {rp*100:>8.2f}% {z_mae:>8.4f} {url}")


if __name__ == "__main__":
    main()
