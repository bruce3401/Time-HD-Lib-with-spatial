#!/usr/bin/env python3
"""F4 — Spatial coherence: plot RegionFormer's learned partition on the CA map.

Steps:
  1. Load a trained RegionFormer checkpoint.
  2. Build a sample batch from the dataset.
  3. Forward pass; extract `model.encoder.layers[0].attention.last_W`.
  4. Argmax over r → integer region label per node.
  5. Load station coordinates from the channel's meta CSV.
  6. Plot CA outline + stations coloured by region label.

Output: manuscripts/Figures/fig_F4_spatial_coherence.pdf
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.RegionFormer import Model as RFModel


CHANNEL_TO_PATHS = {
    "pm25": dict(
        meta="/data/timehd/CalGeo/AirQuality/pm25_meta.csv",
        flow="/data/timehd/CalGeo/AirQuality/pm25_flow.csv",
        adj="/data/timehd/CalGeo/AirQuality/pm25_adj.npy",
    ),
    "outdoor": dict(
        meta="/data/timehd/Mobility_CA/processed/outdoor_discretionary_meta.csv",
        flow="/data/timehd/Mobility_CA/processed/outdoor_discretionary_flow.csv",
        adj="/data/timehd/Mobility_CA/processed/outdoor_discretionary_adj.npy",
    ),
}


def find_ckpt(channel):
    """Most recent ablation_<channel>_R2_softW checkpoint (the paper's §6.2 reference)."""
    if channel == "pm25":
        prefix = "long_term_forecast_RegionFormer_CalGeo-AirQuality-pm25"
    elif channel == "outdoor":
        prefix = "long_term_forecast_RegionFormer_Mobility-CA-outdoor"
    else:
        raise SystemExit(f"unknown channel {channel}")
    base = "/data/checkpoints-timehd"
    for f in sorted(os.listdir(base), reverse=True):
        if f.startswith(prefix) and f"ablation_{channel}_R2_softW_s2021.pth" in f:
            return os.path.join(base, f)
    raise SystemExit("no ablation R2 softW checkpoint found")


def build_model(N, channel):
    """Mirrors R2 config used by sweep_soft_vs_hard.sh (topk K=8) +
    configs/RegionFormer.yaml _calgeo_default."""
    cfg = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=14, pred_len=14, label_len=0,
        d_model=128, d_ff=512, n_heads=4, e_layers=2,
        patch_size=4, patch_stride=2,
        dropout=0.1, embed="timeF", freq="D", activation="gelu",
        factor=3, output_attention=False,
        enc_in=N, dec_in=N, c_out=N, num_class=0,
        # RF-specific (R2 config)
        use_revin=True, revin_no_affine=False,
        r_star=16, gumbel_alpha=0.15,
        use_time_enc=False, grad_ckpt=False, channel_chunk=0,
        use_distance_anchor=True, distance_alpha=1.0,
        scale_mode="topk", topk_within=8, ablate_step=None,
        use_dense_attn=False,
        hard_assignment=False,
        use_coord_embed=True, coord_freqs=8,
        use_graph_prop=True, graph_prop_layers=1,
        use_laplacian_smooth=True, laplacian_lambda=0.05,
        use_adaptive_adj=True, adaptive_adj_dim=32,
        indep_head=False, head_hidden_ratio=4, mlp_head=False,
        use_admin_init=False,
        use_mixer_aux=False, mixer_hidden=64, mixer_blocks=2,
        mixer_gate_init=0.0,
    )
    if channel == "outdoor":
        cfg.seq_len = 16
        cfg.pred_len = 4
    return RFModel(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["pm25", "outdoor"], default="pm25")
    args = ap.parse_args()

    paths = CHANNEL_TO_PATHS[args.channel]
    meta = pd.read_csv(paths["meta"])
    flow = pd.read_csv(paths["flow"], index_col=0)
    adj = np.load(paths["adj"])
    N = flow.shape[1]
    print(f"[{args.channel}] N={N}, meta cols={list(meta.columns)[:8]}")

    ckpt_path = find_ckpt(args.channel)
    print(f"loading {ckpt_path}")
    model = build_model(N, args.channel)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Tolerate missing/unexpected keys (config drift; ablation R might mismatch slightly)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    coords_xy = meta[["lng", "lat"]].values
    if model.use_coord_embed or model.use_graph_prop:
        model.set_spatial_metadata(adj=adj, coord=coords_xy)
    model.eval()

    # Build a single forward batch from the test split (last seq_len rows)
    arr = flow.values.astype(np.float32)
    L = 14 if args.channel == "pm25" else 16
    x = torch.from_numpy(arr[-L:].T[None].transpose(0, 2, 1).copy())  # (1, L, N)
    x_mark = torch.zeros(1, L, 4)
    with torch.no_grad():
        _ = model(x, x_mark, None, None)

    W = model.encoder.layers[0].attention.last_W   # (B, H, r, N)
    print(f"W shape: {tuple(W.shape)}")
    W_mean = W.mean(dim=(0, 1)).cpu().numpy()      # (r, N)
    region = W_mean.argmax(axis=0)                  # (N,)

    # Plot
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    cmap = plt.get_cmap("tab20" if W_mean.shape[0] > 10 else "tab10")
    sc = ax.scatter(coords_xy[:, 0], coords_xy[:, 1],
                    c=region, cmap=cmap, s=24,
                    edgecolors="black", linewidths=0.3, alpha=0.9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Learned region partition — {args.channel} "
                 f"(N={N}, r={W_mean.shape[0]})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    out = Path(f"/data/figures-out/fig_F4_spatial_coherence_{args.channel}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)

    # Also dump the assignment vector for cross-checking.
    summary = {
        "channel": args.channel,
        "N": int(N),
        "r": int(W_mean.shape[0]),
        "region_per_node": region.tolist(),
        "ckpt": ckpt_path,
    }
    Path(f"/data/figures-out/fig_F4_spatial_coherence_{args.channel}.json").write_text(
        json.dumps(summary))


if __name__ == "__main__":
    main()
