#!/usr/bin/env python3
"""Extract RegionFormer cluster argmax for all 8 R2_softW checkpoints.

Reads each ckpt, runs forward pass on the most recent seq_len window, dumps
{channel, N, r, ckpt, region_per_node, lat, lng, ids} to a JSON in
/data/figures-out/case1_W_<channel>.json so the host can build figures.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.RegionFormer import Model as RFModel


CHANNELS = [
    # (key, dataset_dir, file_prefix, sl, pl, calgeo)
    ("pm25",      "CalGeo/AirQuality",       "pm25",                 14, 14, True),
    ("ghi",       "CalGeo/Solar",            "ghi",                  14, 14, True),
    ("tmax",      "CalGeo/Weather",          "tmax",                 14, 14, True),
    ("prcp",      "CalGeo/Weather",          "prcp",                 14, 14, True),
    ("outdoor",   "Mobility_CA/processed",   "outdoor_discretionary",16,  4, False),
    ("essential", "Mobility_CA/processed",   "essential_retail",     16,  4, False),
    ("indoor",    "Mobility_CA/processed",   "indoor_discretionary", 16,  4, False),
    ("food",      "Mobility_CA/processed",   "food_and_drink",        16,  4, False),
]

CKPT_BASE = "/data/checkpoints-timehd"
OUT_DIR = Path("/data/figures-out")
DATA_BASE = "/data/timehd"


def find_ckpt(channel):
    """Locate ablation_<channel>_R2_softW_s2021 ckpt."""
    for f in sorted(os.listdir(CKPT_BASE), reverse=True):
        if f.endswith(f"ablation_{channel}_R2_softW_s2021.pth"):
            return os.path.join(CKPT_BASE, f)
    raise SystemExit(f"no R2 softW ckpt for {channel}")


def build_model(N, sl, pl, calgeo):
    """R2 config from sweep_soft_vs_hard.sh.
    CalGeo channels: d_model=128, n_heads=4, d_ff=512 (calgeo_default).
    Mobility channels: d_model=256, n_heads=8, d_ff=1024 (mobility_default).
    """
    if calgeo:
        d_model, n_heads, d_ff = 128, 4, 512
    else:
        d_model, n_heads, d_ff = 256, 8, 1024
    cfg = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=sl, pred_len=pl, label_len=0,
        d_model=d_model, d_ff=d_ff, n_heads=n_heads, e_layers=2,
        patch_size=4, patch_stride=2,
        dropout=0.1, embed="timeF", freq="D", activation="gelu",
        factor=3, output_attention=False,
        enc_in=N, dec_in=N, c_out=N, num_class=0,
        # RF-specific (R2 config; matches plot_spatial_coherence.py)
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
    return RFModel(cfg)


def extract_one(key, dataset_dir, prefix, sl, pl, calgeo):
    print(f"\n=== {key} ===", flush=True)
    meta_path = f"{DATA_BASE}/{dataset_dir}/{prefix}_meta.csv"
    flow_path = f"{DATA_BASE}/{dataset_dir}/{prefix}_flow.csv"
    adj_path = f"{DATA_BASE}/{dataset_dir}/{prefix}_adj.npy"

    meta = pd.read_csv(meta_path)
    flow = pd.read_csv(flow_path, index_col=0)
    adj = np.load(adj_path)
    N = flow.shape[1]
    print(f"  N={N}, meta cols={list(meta.columns)[:6]}", flush=True)

    ckpt = find_ckpt(key)
    print(f"  ckpt={os.path.basename(ckpt)}", flush=True)

    model = build_model(N, sl, pl, calgeo)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"  load: missing={len(miss)} unexpected={len(unexp)}", flush=True)

    coords_xy = meta[["lng", "lat"]].values.astype(np.float32)
    if model.use_coord_embed or model.use_graph_prop:
        model.set_spatial_metadata(adj=adj, coord=coords_xy)
    model.eval()

    arr = flow.values.astype(np.float32)
    x = torch.from_numpy(arr[-sl:].T[None].transpose(0, 2, 1).copy())
    x_mark = torch.zeros(1, sl, 4)
    with torch.no_grad():
        _ = model(x, x_mark, None, None)

    W = model.encoder.layers[0].attention.last_W   # (B, H, r, N)
    W_mean = W.mean(dim=(0, 1)).cpu().numpy()      # (r, N)
    region = W_mean.argmax(axis=0).astype(int)
    print(f"  W shape={tuple(W.shape)} -> r={W_mean.shape[0]}, "
          f"clusters used={len(set(region.tolist()))}", flush=True)

    id_col = "GEOID" if not calgeo else "site_id"
    ids = meta[id_col].astype(str).tolist()

    out = {
        "channel": key,
        "N": int(N),
        "r": int(W_mean.shape[0]),
        "ckpt": ckpt,
        "region_per_node": region.tolist(),
        "lat": meta["lat"].astype(float).tolist(),
        "lng": meta["lng"].astype(float).tolist(),
        "ids": ids,
        "id_col": id_col,
        "calgeo": bool(calgeo),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"case1_W_{key}.json"
    out_path.write_text(json.dumps(out))
    print(f"  wrote {out_path}", flush=True)


def main():
    for spec in CHANNELS:
        try:
            extract_one(*spec)
        except Exception as e:
            print(f"  ERROR on {spec[0]}: {e}", flush=True)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
