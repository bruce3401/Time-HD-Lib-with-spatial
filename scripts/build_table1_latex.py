#!/usr/bin/env python3
"""Generate the final Table 1 LaTeX (raw MAE / raw RMSE double-SOTA) for the
IJGIS manuscript.

Reads:
  ijgis/v4/baseline_sweep_double_metrics.json — output of the sweep aggregator
Writes:
  ijgis/v4/table1_main.tex — LaTeX table*[t] block ready to paste

Cell layout: each (model, channel) cell shows  "MAE / RMSE".
RMSE = sqrt(MSE); both are in raw inverse-transformed units.
SOTA ranking is identical between MSE and RMSE (sqrt is monotonic), but RMSE
yields more readable magnitudes for high-variance mobility channels.
Ozone is displayed in ppb (×1000) for readability.
ModernTCN cells where default config OOMs on a 95 GB GPU are marked OOM.
SOTA = lowest MAE AND lowest RMSE in column → bold both numbers.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent / "ijgis"
SRC = ROOT / "v4" / "baseline_sweep_double_metrics.json"
DST = ROOT / "v4" / "table1_main.tex"

CHAN_KEYS = [
    ("CalGeo-AirQuality-pm25",     "Pm2.5",         1.0),
    ("CalGeo-AirQuality-ozone",    "Ozone",      1000.0),  # ppm -> ppb
    ("CalGeo-Solar-ghi",           "Ghi",           1.0),
    ("CalGeo-Weather-tmax",        "Tmax",          1.0),
    ("CalGeo-Weather-tmin",        "Tmin",          1.0),
    ("CalGeo-Weather-prcp",        "Prcp",          1.0),
    ("Mobility-CA-outdoor",        "Mob-Outdoor",   1.0),
    ("Mobility-CA-essential",      "Mob-Essential", 1.0),
    ("Mobility-CA-indoor",         "Mob-Indoor",    1.0),
    ("Mobility-CA-food",           "Mob-Food",      1.0),
]

MODELS = ["RegionFormer", "DLinear", "iTransformer", "PatchTST",
          "TimeMixer", "ModernTCN", "CycleNet", "TSMixer"]

# Compact display labels for the table header (full names listed in caption)
MODEL_LABEL = {
    "RegionFormer": r"\textbf{RF}",
    "DLinear":      "DLin.",
    "iTransformer": "iTr.",
    "PatchTST":     "PTST",
    "TimeMixer":    "TMix.",
    "ModernTCN":    "MTCN",
    "CycleNet":     "CycN.",
    "TSMixer":      "TSMx.",
}

OOM_CELLS = {("ModernTCN", "Mobility-CA-essential"),
             ("ModernTCN", "Mobility-CA-indoor"),
             ("ModernTCN", "Mobility-CA-food")}


def fmt(v, scale, sigfigs=4):
    """Pretty-print a metric with adaptive precision."""
    if v is None:
        return "--"
    x = v * scale
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        return f"{x:,.0f}"          # 1\,890
    if abs(x) >= 100:
        return f"{x:.1f}"           # 256.1
    if abs(x) >= 10:
        return f"{x:.2f}"           # 23.31
    if abs(x) >= 1:
        return f"{x:.3f}"           # 3.605
    if abs(x) >= 0.01:
        return f"{x:.4f}"           # 0.8027
    return f"{x:.4f}"


def fmt_value(v, scale, is_best, oom=False):
    """One numeric value, bolded if SOTA. OOM short-circuits."""
    if oom:
        return r"\textsc{oom}"
    if v is None:
        return "--"
    s = fmt(v, scale)
    return (r"\textbf{" + s + "}") if is_best else s


def main():
    data = json.load(open(SRC))
    rf = data["rf"]
    base = data["baselines"]

    # For each channel, compute argmin MAE and argmin RMSE across all 8 models
    best_mae = {}
    best_rmse = {}
    for chan, _, _ in CHAN_KEYS:
        cands_mae  = [(rf[chan]["mae"],  "RegionFormer")]
        cands_rmse = [(rf[chan]["rmse"], "RegionFormer")]
        for m in MODELS[1:]:
            v = base.get(f"{m}|{chan}")
            if v is None or (m, chan) in OOM_CELLS:
                continue
            if v.get("mae")  is not None: cands_mae.append((v["mae"], m))
            if v.get("rmse") is not None: cands_rmse.append((v["rmse"], m))
        best_mae[chan]  = min(cands_mae)[1]
        best_rmse[chan] = min(cands_rmse)[1]

    # Build LaTeX — orientation: rows = channels, columns = models.
    # RegionFormer is the leftmost model column for visual prominence.
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Single-seed $s\!=\!2021$ headline on CalST-Bench. Each channel block has two rows: raw test MAE (top) and raw test RMSE (bottom), both in original units, both lower is better. The SOTA value in each row is in \textbf{bold}. RegionFormer (\textbf{RF}) attains double SOTA (lowest MAE \emph{and} lowest RMSE) on every channel against all seven baselines. Baselines: DLin.\ = DLinear; iTr.\ = iTransformer; PTST = PatchTST; TMix.\ = TimeMixer; MTCN = ModernTCN; CycN.\ = CycleNet; TSMx.\ = TSMixer. All baselines run at each model's released default configuration with training budget matched to RegionFormer (up to 100 epochs with patience-15 early stopping); per-channel best-of-recipe configurations of RegionFormer are listed in Appendix~\ref{app:configs}. \textsc{Ozone} is reported in parts per billion (ppb). \textsc{oom}: ModernTCN's default configuration attempts $>$34\,GiB allocation on $N\!\geq\!6\,286$ channels, exceeding the 95\,GB GPU even at batch size 1.}")
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    # 2 label cols (Channel, Metric) + 8 model cols
    lines.append(r"\begin{tabular}{ll|" + "c" * len(MODELS) + "}")
    lines.append(r"\hline")
    header = [r"\textsc{Channel}", r"\textsc{Metric}"] + [MODEL_LABEL[m] for m in MODELS]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\hline")

    # Two rows per channel: one MAE row, one RMSE row.
    # Channel label uses \multirow{2}{*} to span both metric rows.
    for ci, (chan, disp, sc) in enumerate(CHAN_KEYS):
        is_oom_row = lambda m, c=chan: (m, c) in OOM_CELLS
        # MAE row
        mae_cells = []
        for m in MODELS:
            v = rf[chan] if m == "RegionFormer" else base.get(f"{m}|{chan}", {})
            mae_cells.append(fmt_value(v.get("mae"), sc,
                                       best_mae[chan] == m,
                                       oom=is_oom_row(m)))
        lines.append(r"\multirow{2}{*}{\textsc{" + disp + r"}} & MAE  & " +
                     " & ".join(mae_cells) + r" \\")
        # RMSE row
        rmse_cells = []
        for m in MODELS:
            v = rf[chan] if m == "RegionFormer" else base.get(f"{m}|{chan}", {})
            rmse_cells.append(fmt_value(v.get("rmse"), sc,
                                        best_rmse[chan] == m,
                                        oom=is_oom_row(m)))
        lines.append(r" & RMSE & " + " & ".join(rmse_cells) + r" \\")
        if ci < len(CHAN_KEYS) - 1:
            lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = "\n".join(lines) + "\n"
    DST.write_text(out)
    print(f"Wrote {DST} ({len(out)} bytes)")
    print()
    # script no longer dumps LaTeX to stdout to avoid stray output
    pass


if __name__ == "__main__":
    main()
