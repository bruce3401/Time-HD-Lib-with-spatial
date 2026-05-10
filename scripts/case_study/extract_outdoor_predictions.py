#!/usr/bin/env python3
"""Case 2 — Mobility-CA-outdoor sliding-window predictions over the test split.

Loads the Mobility-CA-outdoor R2_softW_s2021 RegionFormer checkpoint, runs the
forecaster on every test-split window, and saves per-window predictions and
ground truth in raw (unnormalized) space.

Output: /data/figures-out/case2_outdoor_predictions.npz
  - pred_raw (T_win, pl, N)        forecaster output, inverse-scaler
  - true_raw (T_win, pl, N)        held-out values, inverse-scaler
  - input_end_date (T_win,) datetime64[ns]   last date of each input window
  - target_dates (T_win, pl) datetime64[ns]  predicted dates per window
  - GEOID (N,)                      tract IDs
  - lat (N,) / lng (N,)             centroid coordinates
  - county_fips (N,)
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.RegionFormer import Model as RFModel


DATA_DIR = "/data/timehd/Mobility_CA/processed"
CKPT = "/data/checkpoints-timehd/long_term_forecast_RegionFormer_Mobility-CA-outdoor_sl16_pl4_ablation_outdoor_R2_softW_s2021.pth"
OUT = Path("/data/figures-out/case2_outdoor_predictions.npz")
META_OUT = Path("/data/figures-out/case2_outdoor_meta.json")


def build_model(N, sl=16, pl=4):
    """R2 config — Mobility-CA uses d_model=256, n_heads=8, d_ff=1024."""
    cfg = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=sl, pred_len=pl, label_len=0,
        d_model=256, d_ff=1024, n_heads=8, e_layers=2,
        patch_size=4, patch_stride=2,
        dropout=0.1, embed="timeF", freq="D", activation="gelu",
        factor=3, output_attention=False,
        enc_in=N, dec_in=N, c_out=N, num_class=0,
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


def main():
    sl, pl = 16, 4
    SPLIT_R = (0.7, 0.1, 0.2)
    BATCH = 4

    flow_path = f"{DATA_DIR}/outdoor_discretionary_flow.csv"
    meta_path = f"{DATA_DIR}/outdoor_discretionary_meta.csv"
    adj_path = f"{DATA_DIR}/outdoor_discretionary_adj.npy"

    df = pd.read_csv(flow_path, dtype=str)
    df["date"] = pd.to_datetime(df["date"])
    cols = [c for c in df.columns if c != "date"]
    df[cols] = df[cols].astype(np.float32)
    df = df.sort_values("date").reset_index(drop=True)
    flow = df[cols].to_numpy(dtype=np.float32)
    stamp = df["date"].to_numpy(dtype="datetime64[ns]")
    T, N = flow.shape
    print(f"T={T} (rows {stamp[0]} … {stamp[-1]}), N={N}", flush=True)

    meta = pd.read_csv(meta_path, dtype={"GEOID": str, "county_fips": str})
    meta = meta.set_index("GEOID").reindex(cols).reset_index()
    adj = np.load(adj_path).astype(np.float32)

    # Splits
    n_train = int(T * SPLIT_R[0])
    n_val = int(T * SPLIT_R[1])
    test_start_row = T - (T - n_train - n_val)  # = n_train + n_val
    print(f"n_train={n_train} ({stamp[n_train-1].astype('datetime64[D]')}); "
          f"test starts at row {test_start_row} ({stamp[test_start_row]})", flush=True)

    # Scaler fit on train
    scaler = StandardScaler()
    scaler.fit(flow[:n_train])
    data = scaler.transform(flow).astype(np.float32)

    # Test split window starts: data row index `b1` to `b2 - sl - pl + 1`
    # mobility_ca.py border1s[2] = n_total - (n_total - n_train - n_val) - sl
    b1 = test_start_row - sl
    b2 = T
    rel_starts = list(range(0, (b2 - b1) - sl - pl + 1))
    print(f"test windows: {len(rel_starts)} (relative starts 0..{rel_starts[-1]})", flush=True)

    test_data = data[b1:b2]            # (T_test_inflow, N)
    test_stamp = stamp[b1:b2]

    # Model
    print(f"Loading {os.path.basename(CKPT)}", flush=True)
    model = build_model(N, sl, pl)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"  load: missing={len(miss)} unexpected={len(unexp)}", flush=True)

    coords_xy = meta[["lng", "lat"]].values.astype(np.float32)
    if model.use_coord_embed or model.use_graph_prop:
        model.set_spatial_metadata(adj=adj, coord=coords_xy)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Run sliding-window inference in mini-batches
    pred_all_norm = []
    true_all_norm = []
    input_end_dates = []
    target_dates = []

    with torch.no_grad():
        for chunk_start in range(0, len(rel_starts), BATCH):
            idxs = rel_starts[chunk_start:chunk_start + BATCH]
            xs = []
            ys = []
            xs_mark = []
            ys_mark = []
            for i in idxs:
                s_begin = i
                s_end = s_begin + sl
                r_begin = s_end
                r_end = r_begin + pl
                xs.append(test_data[s_begin:s_end])
                ys.append(test_data[r_begin:r_end])
                xs_mark.append(np.zeros((sl, 4), dtype=np.float32))
                ys_mark.append(np.zeros((pl, 4), dtype=np.float32))
                input_end_dates.append(test_stamp[s_end - 1])
                target_dates.append(test_stamp[r_begin:r_end].copy())
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)
            xm = torch.from_numpy(np.stack(xs_mark)).to(device)
            ym = torch.from_numpy(np.stack(ys_mark)).to(device)
            out = model(x, xm, None, ym)         # (B, pl, N)
            pred_all_norm.append(out.cpu().numpy())
            true_all_norm.append(y.cpu().numpy())
            if chunk_start % 16 == 0:
                print(f"  windows {chunk_start}/{len(rel_starts)}", flush=True)

    pred_norm = np.concatenate(pred_all_norm, axis=0)   # (T_win, pl, N)
    true_norm = np.concatenate(true_all_norm, axis=0)
    print(f"pred_norm shape={pred_norm.shape}", flush=True)

    # Inverse-transform back to raw scale (per-feature)
    Tw = pred_norm.shape[0]
    pred_raw = scaler.inverse_transform(pred_norm.reshape(-1, N)).reshape(Tw, pl, N).astype(np.float32)
    true_raw = scaler.inverse_transform(true_norm.reshape(-1, N)).reshape(Tw, pl, N).astype(np.float32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        pred_raw=pred_raw,
        true_raw=true_raw,
        input_end_date=np.array(input_end_dates, dtype="datetime64[ns]"),
        target_dates=np.array(target_dates, dtype="datetime64[ns]"),
        GEOID=meta["GEOID"].to_numpy(),
        lat=meta["lat"].to_numpy(dtype=np.float32),
        lng=meta["lng"].to_numpy(dtype=np.float32),
        county_fips=meta["county_fips"].to_numpy(),
    )
    print(f"Wrote {OUT}", flush=True)
    META_OUT.write_text(json.dumps({
        "ckpt": CKPT,
        "sl": sl, "pl": pl,
        "N": int(N),
        "T": int(T),
        "test_start_row": int(test_start_row),
        "test_start_date": str(stamp[test_start_row]),
        "n_test_windows": int(Tw),
        "first_input_end_date": str(input_end_dates[0]),
        "last_target_end_date": str(target_dates[-1][-1]),
    }, indent=2))
    print(f"Wrote {META_OUT}", flush=True)


if __name__ == "__main__":
    main()
