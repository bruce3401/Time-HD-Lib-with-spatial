#!/usr/bin/env python3
"""Plot LRA scaling benchmark — 2 figures, 4 curves each.

Reads ijgis/v4/scaling_benchmark.json (output of benchmark_lra_scaling.py)
and writes:
  manuscripts/Figures/fig_F3a_scaling_memory.png  (300 DPI)
  manuscripts/Figures/fig_F3b_scaling_flops.png   (300 DPI)

Layout: log-log axes, x = N, y = peak GPU memory (MiB) / FLOPs.
OOM cells are drawn as a red `x` at the last successful N+1 marker so the
reader sees where full attention falls off the cliff.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "ijgis" / "v4" / "scaling_benchmark.json"
DST_DIR = ROOT.parent / "ijgis" / "manuscripts" / "Figures"

CONFIGS = ["full", "topk_K32", "topk_K8"]
LABELS = {
    "full":     r"Full attention (iTransformer) — $\mathcal{O}(N^2)$",
    "topk_K32": r"RegionFormer top-$K$, $K=32$ — $\mathcal{O}(NK)$",
    "topk_K8":  r"RegionFormer top-$K$, $K=8$  — $\mathcal{O}(NK)$",
}
COLORS = {"full": "#D62728", "topk_K32": "#1F77B4", "topk_K8": "#2CA02C"}
MARKERS = {"full": "s", "topk_K32": "o", "topk_K8": "^"}


def load_rows():
    return json.loads(SRC.read_text())


def split_by_config(rows):
    by_cfg = {c: {"N": [], "mib": [], "flops": [], "oom_at": None}
              for c in CONFIGS}
    for r in rows:
        cfg = r["config"]
        if cfg not in by_cfg:
            continue  # skip configs not in CONFIGS (e.g., legacy soft_rN8)
        if r["status"] == "ok":
            by_cfg[cfg]["N"].append(r["N"])
            by_cfg[cfg]["mib"].append(r["peak_mib"])
            by_cfg[cfg]["flops"].append(r["flops"])
        elif r["status"] == "OOM" and by_cfg[cfg]["oom_at"] is None:
            by_cfg[cfg]["oom_at"] = r["N"]
    return by_cfg


def plot_metric(by_cfg, ymetric, ylabel, dst, ymin=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for cfg in CONFIGS:
        d = by_cfg[cfg]
        if d["N"]:
            ax.plot(d["N"], d[ymetric], marker=MARKERS[cfg], color=COLORS[cfg],
                    label=LABELS[cfg], linewidth=1.5, markersize=6)
        if d["oom_at"] is not None and d[ymetric]:
            ax.plot([d["oom_at"]], [d[ymetric][-1] * 1.6],
                    marker="x", color=COLORS[cfg],
                    markersize=10, markeredgewidth=2.5, linestyle="None")
            ax.annotate("OOM", xy=(d["oom_at"], d[ymetric][-1] * 1.6),
                        xytext=(6, 0), textcoords="offset points",
                        color=COLORS[cfg], fontsize=8, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel(r"Number of variates $N$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    fig.tight_layout()
    DST_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, format="png", dpi=300, bbox_inches="tight")
    print(f"Wrote {dst}")
    plt.close(fig)


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; run benchmark_lra_scaling.py first")
    rows = load_rows()
    by_cfg = split_by_config(rows)
    plot_metric(by_cfg, "mib",
                "Peak GPU memory (MiB)",
                DST_DIR / "fig_F3a_scaling_memory.png")
    plot_metric(by_cfg, "flops",
                "Forward FLOPs (analytical)",
                DST_DIR / "fig_F3b_scaling_flops.png")


if __name__ == "__main__":
    main()
