"""DCRNN — Diffusion Convolutional Recurrent Neural Network (Li, Yu, Shahabi & Liu, ICLR 2018).

Canonical encoder-decoder GRU where every linear inside each gate is replaced by
a K-step diffusion convolution over two random-walk supports:

    DConv(X) = sum_{k=0}^{K} (P_fwd^k X) Theta^fwd_k + (P_bwd^k X) Theta^bwd_k
    P_fwd = D_O^{-1} A,    P_bwd = D_I^{-1} A^T

The graph is *fixed* (not adaptive) — the canonical setting that the IJGIS
reviewer asked us to compare against on the same derived adjacencies (KNN /
queen contiguity) used by RegionFormer.

This implementation:
  * applies P_fwd and P_bwd as sparse-dense matmuls so we scale to N ~ 10k
    (mob-food has N=8474; queen contiguity is very sparse);
  * uses free-running decoding (no teacher forcing / scheduled sampling) — this
    is the simplification adopted by most subsequent reimplementations and
    avoids needing y at forward() time;
  * does its own per-channel z-score normalisation in the forward pass, mirror
    of the SpatialLCA / RegionFormer convention.

Hyper-defaults (from the original paper): K=2 diffusion steps, hidden=64,
2 encoder + 2 decoder layers.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from core.registry import register_model


def _sparse_dense_bmm(A_sparse: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A_sparse: (N, N) sparse_coo. x: (B, N, F). Returns (B, N, F).

    Implemented as one sparse-mm: (N, N) @ (N, B*F) → (N, B*F)."""
    B, N, F = x.shape
    xt = x.permute(1, 0, 2).reshape(N, B * F)
    yt = torch.sparse.mm(A_sparse, xt)
    return yt.reshape(N, B, F).permute(1, 0, 2).contiguous()


def _diffusion_conv(
    x: torch.Tensor,
    P_fwd: torch.Tensor,
    P_bwd: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """DCRNN's K-step bidirectional diffusion conv applied to (B, N, F).

    Returns (B, N, (1 + 2K) * F), the concatenation of P^0 X (=X), P_fwd^1..K X,
    and P_bwd^1..K X along the feature axis. The Theta projection is applied by
    the caller (a Linear)."""
    outs = [x]
    cur_fwd = x
    cur_bwd = x
    for _ in range(K):
        cur_fwd = _sparse_dense_bmm(P_fwd, cur_fwd)
        cur_bwd = _sparse_dense_bmm(P_bwd, cur_bwd)
        outs.append(cur_fwd)
        outs.append(cur_bwd)
    return torch.cat(outs, dim=-1)


class DCGRUCell(nn.Module):
    """Diffusion convolutional GRU cell — gates and candidate use DConv instead of Linear."""

    def __init__(self, num_nodes: int, in_dim: int, hid_dim: int, K: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.K = K
        # Each DConv expands feature axis by (1 + 2K). Gates and candidate take [x; h].
        n_supports = 1 + 2 * K
        feat_in = n_supports * (in_dim + hid_dim)
        self.gate_proj = nn.Linear(feat_in, 2 * hid_dim)  # r and u packed
        self.cand_proj = nn.Linear(feat_in, hid_dim)
        # Init biases to encourage stable training (gate biases positive → gates open).
        nn.init.constant_(self.gate_proj.bias, 1.0)
        nn.init.constant_(self.cand_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,    # (B, N, in_dim)
        h: torch.Tensor,    # (B, N, hid_dim)
        P_fwd: torch.Tensor,
        P_bwd: torch.Tensor,
    ) -> torch.Tensor:
        xh = torch.cat([x, h], dim=-1)  # (B, N, in_dim + hid_dim)
        gconv = _diffusion_conv(xh, P_fwd, P_bwd, self.K)
        gates = torch.sigmoid(self.gate_proj(gconv))
        r, u = gates.chunk(2, dim=-1)
        xrh = torch.cat([x, r * h], dim=-1)
        cconv = _diffusion_conv(xrh, P_fwd, P_bwd, self.K)
        c = torch.tanh(self.cand_proj(cconv))
        return u * h + (1.0 - u) * c


class DCRNNEncoder(nn.Module):
    """L-layer DCGRU encoder. Output: final hidden states per layer, (L, B, N, hid)."""

    def __init__(self, num_nodes: int, in_dim: int, hid_dim: int, num_layers: int, K: int):
        super().__init__()
        self.num_layers = num_layers
        cells = []
        for i in range(num_layers):
            cells.append(DCGRUCell(num_nodes, in_dim if i == 0 else hid_dim, hid_dim, K))
        self.cells = nn.ModuleList(cells)
        self.num_nodes = num_nodes
        self.hid_dim = hid_dim

    def forward(
        self,
        x_seq: torch.Tensor,   # (B, T, N, in_dim)
        P_fwd: torch.Tensor,
        P_bwd: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, _ = x_seq.shape
        device = x_seq.device
        hs = [torch.zeros(B, N, self.hid_dim, device=device) for _ in range(self.num_layers)]
        for t in range(T):
            inp = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                hs[i] = cell(inp, hs[i], P_fwd, P_bwd)
                inp = hs[i]
        return torch.stack(hs, dim=0)  # (L, B, N, hid)


class DCRNNDecoder(nn.Module):
    """L-layer DCGRU decoder with a per-step output head. Free-running (feeds its own prediction)."""

    def __init__(self, num_nodes: int, out_dim: int, hid_dim: int, num_layers: int, K: int):
        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim
        cells = []
        for i in range(num_layers):
            cells.append(DCGRUCell(num_nodes, out_dim if i == 0 else hid_dim, hid_dim, K))
        self.cells = nn.ModuleList(cells)
        self.proj = nn.Linear(hid_dim, out_dim)

    def forward(
        self,
        h_init: torch.Tensor,   # (L, B, N, hid)
        pred_len: int,
        P_fwd: torch.Tensor,
        P_bwd: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = h_init.shape[1], h_init.shape[2], h_init.shape[3]
        device = h_init.device
        hs = [h_init[i] for i in range(self.num_layers)]
        # GO symbol: zeros at first step.
        inp = torch.zeros(B, N, self.out_dim, device=device)
        outs = []
        for _ in range(pred_len):
            x = inp
            for i, cell in enumerate(self.cells):
                hs[i] = cell(x, hs[i], P_fwd, P_bwd)
                x = hs[i]
            y = self.proj(x)  # (B, N, out_dim)
            outs.append(y)
            inp = y  # free-running
        return torch.stack(outs, dim=1)  # (B, pred_len, N, out_dim)


@register_model("DCRNN", paper="Diffusion Convolutional Recurrent Neural Network (Li et al. ICLR 2018)", year=2018)
class Model(nn.Module):
    """DCRNN baseline. Per-channel z-norm in/out wrapper around an encoder-decoder DCGRU."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Hyperparameters: respect d_model if set in user config, else canonical 64.
        self.hid_dim = int(getattr(configs, "d_model", 64))
        self.num_layers = int(getattr(configs, "e_layers", 2))
        self.K = int(getattr(configs, "diffusion_K", 2))
        self.in_dim = 1   # univariate per-node input feature
        self.out_dim = 1
        self.encoder = DCRNNEncoder(self.enc_in, self.in_dim, self.hid_dim, self.num_layers, self.K)
        self.decoder = DCRNNDecoder(self.enc_in, self.out_dim, self.hid_dim, self.num_layers, self.K)
        # Sparse supports populated by set_spatial_metadata. Empty placeholders → forward()
        # will raise if metadata wasn't injected (caught by experiment harness).
        self.register_buffer(
            "_P_fwd_indices", torch.empty(2, 0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_P_fwd_values", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_P_bwd_indices", torch.empty(2, 0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_P_bwd_values", torch.empty(0, dtype=torch.float32), persistent=False
        )
        print(f"[DCRNN] enc_in={self.enc_in} | seq_len={self.seq_len} | pred_len={self.pred_len} | "
              f"hid={self.hid_dim} | layers={self.num_layers} | K={self.K}")

    def set_spatial_metadata(self, adj, coord=None, admin_groups=None):
        """Build P_fwd = D_O^{-1} A and P_bwd = D_I^{-1} A^T as sparse_coo buffers.

        Self-loops are added before normalisation (DCRNN convention)."""
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().cpu().numpy()
        adj = np.asarray(adj, dtype=np.float32).copy()
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj must be square 2D, got {adj.shape}")
        N = adj.shape[0]
        # Add self-loop.
        adj = adj + np.eye(N, dtype=np.float32)
        # Forward random-walk: row-normalise A.
        deg_out = adj.sum(axis=1, keepdims=True)
        deg_out[deg_out == 0] = 1.0
        P_fwd = adj / deg_out
        # Backward random-walk: row-normalise A^T (= column-normalise A).
        adj_T = adj.T
        deg_in = adj_T.sum(axis=1, keepdims=True)
        deg_in[deg_in == 0] = 1.0
        P_bwd = adj_T / deg_in

        def _to_sparse(P):
            rows, cols = np.nonzero(P)
            vals = P[rows, cols]
            idx = torch.from_numpy(np.stack([rows, cols], axis=0).astype(np.int64))
            v = torch.from_numpy(vals.astype(np.float32))
            return idx, v

        f_idx, f_val = _to_sparse(P_fwd)
        b_idx, b_val = _to_sparse(P_bwd)
        device = self._P_fwd_indices.device
        self._P_fwd_indices = f_idx.to(device)
        self._P_fwd_values = f_val.to(device)
        self._P_bwd_indices = b_idx.to(device)
        self._P_bwd_values = b_val.to(device)
        self._N = N
        density = (adj > 0).sum() / float(N * N)
        print(f"[DCRNN] supports built: N={N}, density={density:.4f}, "
              f"|E_fwd|={f_val.numel()}, |E_bwd|={b_val.numel()}")

    def _supports(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise the sparse_coo P_fwd and P_bwd from buffers."""
        if self._P_fwd_values.numel() == 0:
            raise RuntimeError(
                "DCRNN requires set_spatial_metadata() to be called before forward(). "
                "The experiment harness should auto-inject this when train_data has .adj."
            )
        N = self._N
        P_fwd = torch.sparse_coo_tensor(
            self._P_fwd_indices, self._P_fwd_values, size=(N, N),
            device=self._P_fwd_values.device,
        ).coalesce()
        P_bwd = torch.sparse_coo_tensor(
            self._P_bwd_indices, self._P_bwd_values, size=(N, N),
            device=self._P_bwd_values.device,
        ).coalesce()
        return P_fwd, P_bwd

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: (B, T, N) — per-channel z-score normalise inside forward (RevIN-lite).
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev

        # Reshape to DCRNN's (B, T, N, F=1).
        x_seq = x_norm.unsqueeze(-1)
        P_fwd, P_bwd = self._supports()
        h_final = self.encoder(x_seq, P_fwd, P_bwd)             # (L, B, N, hid)
        out = self.decoder(h_final, self.pred_len, P_fwd, P_bwd)  # (B, pred_len, N, 1)
        out = out.squeeze(-1)                                    # (B, pred_len, N)

        # Denormalise back to original scale.
        return out * stdev + means
