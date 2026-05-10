#!/usr/bin/env python3
"""LRA scaling benchmark — forward-only memory + FLOPs sweep.

For 4 model configs and 8 N values, build a fresh model, run a synthetic
forward pass, and record peak GPU memory + analytical FLOPs.

Configs:
  full   : iTransformer (full self-attention over N variates)         O(N^2)
  rN8    : RegionFormer with r = max(8, N//8)  (proportional)         O(N^2/8)
  r32    : RegionFormer with r = 32 (constant)                        O(Nr + r^2)
  r8     : RegionFormer with r = 8  (extreme low-rank)                ~O(N)

N grid: 64, 128, 256, 512, 1024, 2048, 4096, 8192.

OOM is recorded as None (the curve simply ends).

Output: ijgis/v4/scaling_benchmark.json with rows:
  {config, N, peak_mib, flops, status}
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.iTransformer import Model as ITrModel
from models.RegionFormer import Model as RFModel

OUT = ROOT.parent / "ijgis" / "v4" / "scaling_benchmark.json"
N_GRID = [128, 512, 2048, 8192, 16384, 32768, 65536, 131072]
# 4 configs: full self-attention vs 3 LRA variants exploring the cost-quality knob.
#   full       : iTransformer (full N x N scores)                  O(N^2)
#   soft_rN8   : RF soft within-region with r = N/8                O(N^2) but ~1/8 const
#   topk_K8    : RF top-K sparse within-region, K=8                O(N K)
#   topk_K32   : RF top-K sparse within-region, K=32               O(N K)
CONFIGS = ["full", "soft_rN8", "topk_K8", "topk_K32"]
SEQ_LEN = 14
PRED_LEN = 14
D_MODEL = 64
N_HEADS = 4
E_LAYERS = 1   # single layer keeps memory budget for full attention to higher N
D_FF = 256
DROPOUT = 0.1


def base_configs(N):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=SEQ_LEN, pred_len=PRED_LEN, label_len=0,
        d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS, e_layers=E_LAYERS,
        dropout=DROPOUT, embed="timeF", freq="d", activation="gelu",
        factor=3, output_attention=False,
        enc_in=N, dec_in=N, c_out=N, num_class=0,
    )


def make_itr(N):
    cfg = base_configs(N)
    return ITrModel(cfg)


def make_rf(N, r_value, scale_mode="soft", topk_within=8):
    cfg = base_configs(N)
    # RegionFormer-specific knobs (mirror argparse defaults)
    cfg.use_revin = True
    cfg.revin_no_affine = False
    cfg.r_star = r_value
    cfg.gumbel_alpha = 0.0
    cfg.patch_size = 1
    cfg.patch_stride = 1
    cfg.use_time_enc = False
    cfg.grad_ckpt = False
    cfg.channel_chunk = 0
    cfg.use_distance_anchor = False
    cfg.distance_alpha = 0.0
    cfg.scale_mode = scale_mode
    cfg.topk_within = topk_within
    cfg.ablate_step = None
    cfg.use_dense_attn = False
    cfg.use_coord_embed = False
    cfg.coord_freqs = 8
    cfg.use_graph_prop = False
    cfg.graph_prop_layers = 0
    cfg.use_laplacian_smooth = False
    cfg.laplacian_lambda = 0.0
    cfg.use_adaptive_adj = False
    cfg.adaptive_adj_dim = 8
    cfg.indep_head = False
    cfg.head_hidden_ratio = 4
    cfg.mlp_head = False
    cfg.use_admin_init = False
    cfg.use_mixer_aux = False
    cfg.mixer_hidden = 64
    cfg.mixer_blocks = 2
    cfg.mixer_gate_init = 0.0
    return RFModel(cfg)


def analytical_flops(config, N):
    """Forward-pass FLOPs estimate (1 layer, B=1, ignores embedding/projection
    and FFN since they are equal across configs and dominate by O(N D^2)).
    Counts only the spatial-coupling cost: full QK^T + softmax-V vs LRA."""
    D = D_MODEL
    H = N_HEADS
    d = D // H
    if config == "full":
        return 2 * H * N * N * d
    if config == "soft_rN8":
        r = max(8, N // 8)
        s1 = H * N * r * D                 # assignment
        s2 = 2 * H * N * N * d             # within-region soft (still N^2)
        s3 = 2 * H * r * r * d             # cross-region (tiny)
        s4 = H * N * r * d                 # fusion scatter
        return s1 + s2 + s3 + s4
    # top-K variants — true O(N K)
    K = {"topk_K8": 8, "topk_K32": 32}[config]
    r = 8  # constant low rank for both top-K configs
    s1 = H * N * r * D
    s2 = 2 * H * N * K * d                 # genuinely O(N K)
    s3 = 2 * H * r * r * d
    s4 = H * N * r * d
    return s1 + s2 + s3 + s4


def measure(config, N, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if config == "full":
        model = make_itr(N).to(device).eval()
    elif config == "soft_rN8":
        r = max(8, N // 8)
        model = make_rf(N, r, scale_mode="soft").to(device).eval()
    elif config == "topk_K8":
        model = make_rf(N, 8, scale_mode="topk", topk_within=8).to(device).eval()
    elif config == "topk_K32":
        model = make_rf(N, 8, scale_mode="topk", topk_within=32).to(device).eval()
    else:
        raise ValueError(config)
    x_enc = torch.randn(1, SEQ_LEN, N, device=device)
    x_mark = torch.randn(1, SEQ_LEN, 4, device=device)
    try:
        with torch.no_grad():
            y = model(x_enc, x_mark, None, None)
            if isinstance(y, tuple):
                y = y[0]
            torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        peak_mib = None
        status = "OOM"
    except Exception as e:
        peak_mib = None
        status = f"ERR: {type(e).__name__}: {str(e)[:120]}"
    finally:
        del model
        torch.cuda.empty_cache()
    return peak_mib, status


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")
    rows = []
    for cfg in CONFIGS:
        for N in N_GRID:
            print(f"  [{cfg:5s} N={N:5d}] ", end="", flush=True)
            mib, status = measure(cfg, N, device)
            flops = analytical_flops(cfg, N)
            row = {"config": cfg, "N": N, "peak_mib": mib,
                   "flops": flops, "status": status}
            rows.append(row)
            mib_s = f"{mib:8.1f} MiB" if mib is not None else "      OOM"
            print(f"{mib_s}  flops={flops:.2e}  status={status}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
