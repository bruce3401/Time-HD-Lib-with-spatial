"""RegionFormer — latent-region attention with within-region (Step 2) and
cross-region (Step 3) pathways sharing one learned soft assignment.

Design doc: ijgis/v2/architecture.md
Pitch: speed-matched SOTA — raw MAE within ±0.5 of PatchSTG/PASTN at >= 2x
wall-clock speedup vs gwnet, plus IJGIS-legible spatial regions.

Reuses from SpatialLCA: InvertedPatchEmbedding, RevIN, and the assignment
machinery (E1 distance anchor, E3 admin-seeded K-means init, gumbel, log_tau).
The attention pathway itself is new.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import register_model
from .SpatialLCA import (
    LatentCentroidAttention,
    InvertedPatchEmbedding,
    RevIN,
)


class RegionAttention(nn.Module):
    """Steps 1-4 of RegionFormer in one module.

    Step 1: soft region assignment W (B, H, r, N) — reused from LCA.
    Step 2: within-region attention with soft mask M = log(W^T W).
    Step 3: cross-region attention among r region tokens z_r = sum_n W[r,n]*v[n].
    Step 4: fuse — y_n = A_wr_n + sum_r W[r,n] * A_cr_r.
    """

    def __init__(self, d_model, n_heads, r,
                 attn_dropout=0.1, gumbel_alpha=0.15,
                 use_distance_anchor=True, distance_alpha=1.0,
                 scale_mode="soft", topk_within=8,
                 ablate_step=None, use_dense_attn=False,
                 hard_assignment=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads
        self.r = r
        self.d = d_model // n_heads
        self.D = d_model
        self.scale_mode = scale_mode
        self.topk_within = int(topk_within)
        # ablate_step ∈ {None, "within", "cross"}:
        #   "within" → zero A_wr (keep cross-region only ≡ classic LCA)
        #   "cross"  → zero bcast(A_cr) (keep within-region only)
        assert ablate_step in (None, "within", "cross"), f"bad ablate_step={ablate_step!r}"
        self.ablate_step = ablate_step

        # Dense-attention augmentation: parallel iTransformer-style full
        # cross-variate attention path with a learnable scalar gate (init 0,
        # so model behaves like baseline RF; can learn to use it if helpful).
        # In the limit γ→1, RF subsumes iTransformer's attention pathway.
        self.use_dense_attn = bool(use_dense_attn)
        if self.use_dense_attn:
            self.dense_gate = nn.Parameter(torch.tensor(0.0))

        # Reuse LCA for Step 1 machinery (assign_net, log_tau, anchor, gumbel,
        # admin K-means seeding via set_coord_and_init). LCA's q/k/v/out
        # projections inside `self.assign` are unused; keeps E1/E3 logic in
        # one place at the cost of ~D*D*4 dead params.
        self.assign = LatentCentroidAttention(
            d_model, n_heads, r,
            attention_dropout=attn_dropout,
            gumbel_alpha=gumbel_alpha,
            use_distance_anchor=use_distance_anchor,
            distance_alpha=distance_alpha,
        )

        # RegionFormer-specific projections (shared across Steps 2 and 3).
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # Cross-region pathway has its own query (z attends to z).
        self.q_cr = nn.Linear(self.d, self.d)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(attn_dropout)

        self.last_W = None  # populated each forward; consumed by Laplacian loss.
        # Hard-assignment ablation: forward uses one-hot(argmax(W)); backward
        # uses straight-through identity so optimisation still proceeds.
        self.hard_assignment = bool(hard_assignment)

    def _within_region_soft(self, q, k, v, W):
        """Step 2 — soft-mask within-region attention.

        q, k, v: (B, H, N, d); W: (B, H, r, N).  Returns (B, H, N, d).
        Bias M_ij = log( sum_r W[r,i]*W[r,j] ) makes attention attend
        preferentially within shared regions while staying differentiable in W.
        """
        # M = log(W^T W) ∈ (B, H, N, N)
        WtW = torch.einsum("bhrm,bhrn->bhmn", W, W).clamp_min(1e-8)
        M = WtW.log()
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.d)
        scores = scores + M
        attn = self.drop(scores.softmax(dim=-1))
        return torch.einsum("bhij,bhjd->bhid", attn, v)

    def _within_region_topk(self, q, k, v, W, K=8):
        """Step 2 — true-sparse top-K within-region attention.

        For each query i, pick its preferred region r*[i] = argmax_r W[r,i],
        then attend only to the K key/value vectors of that region's top-K
        members. Memory O(B·H·N·K·d) instead of O(B·H·N²) — required for
        large-N channels (mob-essential 6286, mob-indoor 6427, mob-food 8474)
        which OOM on the dense 'soft' path at N²>=80M scores per layer.

        Numerically equivalent to the prior masked-softmax implementation:
        masked positions in dense softmax produce zero attention; here we
        directly compute softmax over only the K kept positions.

        Gradient flow through W: per-region topk and argmax are not
        differentiable, but W is updated via Step 3 cross-region attention
        and the Laplacian regulariser, same as the prior implementation.
        """
        B, H, N, d = q.shape
        K_eff = min(int(K), N)
        # Per-region top-K members, computed once over (B, H, r) — small.
        topk_per_region = W.topk(K_eff, dim=-1).indices                # (B, H, r, K) int64
        # Each query's preferred region.
        r_star = W.argmax(dim=-2)                                      # (B, H, N) int64
        # For each query i, look up its region's K member indices.
        r_star_exp = r_star.unsqueeze(-1).expand(-1, -1, -1, K_eff)    # (B, H, N, K)
        members = torch.gather(topk_per_region, dim=2, index=r_star_exp)  # (B, H, N, K)
        # Gather K key/value vectors per query without ever materialising N×N.
        flat_idx = members.reshape(B, H, N * K_eff)
        flat_idx_d = flat_idx.unsqueeze(-1).expand(B, H, N * K_eff, d)
        k_gat = torch.gather(k, dim=2, index=flat_idx_d).view(B, H, N, K_eff, d)
        v_gat = torch.gather(v, dim=2, index=flat_idx_d).view(B, H, N, K_eff, d)
        # Compute attention only over the K kept positions per query.
        scores = torch.einsum("bhid,bhikd->bhik", q, k_gat) / math.sqrt(self.d)
        attn = self.drop(scores.softmax(dim=-1))
        return torch.einsum("bhik,bhikd->bhid", attn, v_gat)

    def _within_region_grouped(self, q, k, v, W):
        """Step 2 — hard-grouped fallback for CA scale (N=8600).

        Soft mask requires O(B*H*N^2) memory which exceeds 80GB at N=8600.
        Implementation deferred to Phase 3.5; use scale_mode='topk' as a
        capacity-reducing alternative when overfit-prone.
        """
        raise NotImplementedError(
            "scale_mode='grouped' planned for Phase 3.5 (CA scale). "
            "For SD/GBA/GLA use scale_mode='soft' (default) or 'topk'."
        )

    def forward(self, x, attn_mask=None, **kw):
        B, N, D = x.shape
        # ----- Step 1: assignment -----
        W = self.assign._assignment(x)            # (B, H, r, N)
        if self.hard_assignment:
            # Straight-through estimator: forward = onehot(argmax(W)), backward
            # = identity through W. Isolates the value of soft probabilistic
            # mixing in the forward pass while keeping training viable.
            idx = W.argmax(dim=-2, keepdim=True)               # (B, H, 1, N)
            W_hard = torch.zeros_like(W).scatter_(-2, idx, 1.0)
            W = W + (W_hard - W).detach()
        self.last_W = W

        # ----- projections -----
        q = self.q_proj(x).view(B, N, self.H, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.H, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.H, self.d).transpose(1, 2)

        # ----- Step 2: within-region attention -----
        if self.scale_mode == "grouped":
            A_wr = self._within_region_grouped(q, k, v, W)
        elif self.scale_mode == "topk":
            A_wr = self._within_region_topk(q, k, v, W, K=self.topk_within)
        else:
            A_wr = self._within_region_soft(q, k, v, W)

        # ----- Step 3: cross-region attention among r region tokens -----
        z = torch.einsum("bhrn,bhnd->bhrd", W, v)              # (B, H, r, d)
        q_r = self.q_cr(z)
        s_cr = torch.einsum("bhrd,bhsd->bhrs", q_r, z) / math.sqrt(self.d)
        attn_cr = self.drop(s_cr.softmax(dim=-1))
        A_cr = torch.einsum("bhrs,bhsd->bhrd", attn_cr, z)     # (B, H, r, d)

        # ----- Step 4: broadcast cross-region back via W; fuse with within-region -----
        bcast = torch.einsum("bhrn,bhrd->bhnd", W, A_cr)        # (B, H, N, d)
        if self.ablate_step == "within":
            y = bcast                          # cross-region only ≡ classic LCA
        elif self.ablate_step == "cross":
            y = A_wr                           # within-region only
        else:
            y = A_wr + bcast                   # full RegionFormer

        # ----- Optional: dense iTr-style attention path (gated, init 0) -----
        if self.use_dense_attn:
            # Standard scaled dot-product attention across all N variates,
            # no spatial mask — exactly what iTransformer's variate-attention does.
            scores_d = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.d)
            attn_d = self.drop(scores_d.softmax(dim=-1))
            A_dense = torch.einsum("bhij,bhjd->bhid", attn_d, v)   # (B, H, N, d)
            y = y + self.dense_gate * A_dense

        out = y.transpose(1, 2).reshape(B, N, self.D)
        return self.out(out)

class RegionFormerEncoderLayer(nn.Module):
    """Pre-norm transformer block: RegionAttention + FFN."""

    def __init__(self, d_model, n_heads, r, d_ff=None, dropout=0.1,
                 attn_dropout=0.1, gumbel_alpha=0.15,
                 use_distance_anchor=True, distance_alpha=1.0,
                 scale_mode="soft", topk_within=8,
                 ablate_step=None, use_dense_attn=False,
                 hard_assignment=False):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = RegionAttention(
            d_model, n_heads, r,
            attn_dropout=attn_dropout,
            gumbel_alpha=gumbel_alpha,
            use_distance_anchor=use_distance_anchor,
            distance_alpha=distance_alpha,
            scale_mode=scale_mode,
            topk_within=topk_within,
            ablate_step=ablate_step,
            use_dense_attn=use_dense_attn,
            hard_assignment=hard_assignment,
        )
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x)))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, grad_ckpt=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.grad_ckpt = grad_ckpt

    def forward(self, x):
        for layer in self.layers:
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x) if self.norm is not None else x


@register_model(
    "RegionFormer",
    paper="SpatialScale: RegionFormer for Spatially Explicit Time-Series Forecasting",
    year=2026,
)
class Model(nn.Module):
    """RegionFormer top-level model. See ijgis/v2/architecture.md §3."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.use_laplacian_smooth = getattr(configs, "use_laplacian_smooth", False)
        self.laplacian_lambda = float(getattr(configs, "laplacian_lambda", 0.05))

        # Spatial metadata buffers: filled by set_spatial_metadata().
        self.register_buffer("edge_index", torch.empty(2, 0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_weight", torch.empty(0, dtype=torch.float32), persistent=False)

        self.use_distance_anchor = getattr(configs, "use_distance_anchor", True)
        self.distance_alpha = float(getattr(configs, "distance_alpha", 1.0))
        self.scale_mode = getattr(configs, "scale_mode", "soft")
        self.topk_within = int(getattr(configs, "topk_within", 8))

        # Coordinate-aware Fourier-feature embedding (NeRF-style).
        # Information iTransformer cannot use — variates are exchangeable in iTr.
        self.use_coord_embed = bool(getattr(configs, "use_coord_embed", False))
        self.coord_freqs = int(getattr(configs, "coord_freqs", 8))
        # coord_emb buffer is filled by set_spatial_metadata once coords are known.
        self.register_buffer("coord_emb", torch.empty(0, dtype=torch.float32), persistent=False)

        # E4: graph-propagation residual (D^{-1/2} (A+I) D^{-1/2} matmul on enc_in).
        # Ported from SpatialLCA — provides explicit GCN-style message passing
        # over the road/KNN adjacency, complementing the implicit LCA assignment.
        # Gate initialized to 0 so model behaves like baseline at start.
        self.use_graph_prop = bool(getattr(configs, "use_graph_prop", False))
        self.graph_prop_layers = int(getattr(configs, "graph_prop_layers", 1))
        if self.use_graph_prop:
            self.register_buffer("A_hat", torch.empty(0, dtype=torch.float32), persistent=False)
            self.graph_gate = nn.Parameter(torch.tensor(0.0))

        # Adaptive adjacency (low-rank learnable residual on top of fixed A_hat).
        # Implements GraphWaveNet-style learned graph: A_learned = tanh(E1 @ E2.T)
        # where E1, E2 are nn.Embedding(N, K). 2NK params instead of N². Init at
        # 0 (via small init + gate), so model starts behaving like fixed graph.
        # Requires --use_graph_prop to be effective.
        self.use_adaptive_adj = bool(getattr(configs, "use_adaptive_adj", False))
        self.adaptive_adj_dim = int(getattr(configs, "adaptive_adj_dim", 8))
        if self.use_adaptive_adj:
            self.adj_emb1 = nn.Embedding(int(getattr(configs, "enc_in")), self.adaptive_adj_dim)
            self.adj_emb2 = nn.Embedding(int(getattr(configs, "enc_in")), self.adaptive_adj_dim)
            nn.init.normal_(self.adj_emb1.weight, std=0.01)
            nn.init.normal_(self.adj_emb2.weight, std=0.01)
            # Mixing gate between fixed and learned graph: alpha=0 → fixed only.
            self.adaptive_adj_gate = nn.Parameter(torch.tensor(0.0))
        # Head config — mirrors SpatialLCA flags so CLI is consistent.
        # Defaults: mlp_head=False, indep_head=False → single shared nn.Linear (matches SpatialLCA + STAEformer/PatchSTG conventions).
        self.mlp_head = getattr(configs, "mlp_head", False)
        self.indep_head = getattr(configs, "indep_head", False)
        self.head_hidden_ratio = int(getattr(configs, "head_hidden_ratio", 1))

        self.r = configs.r_star
        self.dropout = configs.dropout
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.pred_len = configs.pred_len
        self.gumbel_alpha = configs.gumbel_alpha
        self.use_time_enc = getattr(configs, "use_time_enc", False)
        self.e_layers = configs.e_layers
        self.use_revin = getattr(configs, "use_revin", True)
        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=not getattr(configs, "revin_no_affine", False))
        self.enc_in = configs.enc_in

        # Coord embedding projection: Fourier features [N, 2*2*F] -> [N, d_model]
        if self.use_coord_embed:
            self.coord_proj = nn.Linear(2 * 2 * self.coord_freqs, self.d_model)

        self.enc_embedding = InvertedPatchEmbedding(
            seq_len=configs.seq_len,
            d_model=self.d_model,
            patch_size=configs.patch_size,
            patch_stride=configs.patch_stride,
            dropout=self.dropout,
            use_time_enc=self.use_time_enc,
            keep_patches=False,
            grad_ckpt=getattr(configs, "grad_ckpt", False),
            channel_chunk=getattr(configs, "channel_chunk", 0),
        )

        self.encoder = Encoder(
            [
                RegionFormerEncoderLayer(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    r=self.r,
                    d_ff=configs.d_ff or self.d_model * 4,
                    dropout=self.dropout,
                    attn_dropout=self.dropout,
                    gumbel_alpha=self.gumbel_alpha,
                    use_distance_anchor=self.use_distance_anchor,
                    distance_alpha=self.distance_alpha,
                    scale_mode=self.scale_mode,
                    topk_within=self.topk_within,
                    ablate_step=getattr(configs, "ablate_step", None) or None,
                    use_dense_attn=bool(getattr(configs, "use_dense_attn", False)),
                    hard_assignment=bool(getattr(configs, "hard_assignment", False)),
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
            grad_ckpt=getattr(configs, "grad_ckpt", False),
        )

        # Output head — same flag taxonomy as SpatialLCA (architecture.md §11 Q5).
        #   mlp_head=False, indep_head=False → single shared nn.Linear  ← default
        #   mlp_head=True,  indep_head=False → single shared MLP
        #   mlp_head=*,     indep_head=True  → per-channel ModuleList of heads
        def _build_head():
            if self.mlp_head:
                h_dim = self.d_model * self.head_hidden_ratio
                return nn.Sequential(
                    nn.Linear(self.d_model, h_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(h_dim, self.pred_len),
                )
            return nn.Linear(self.d_model, self.pred_len)

        if self.indep_head:
            self.output = nn.ModuleList([_build_head() for _ in range(self.enc_in)])
        else:
            self.output = _build_head()

        head_tag = ("indep_" if self.indep_head else "shared_") + ("mlp" if self.mlp_head else "linear")
        ablate = getattr(configs, "ablate_step", None) or "none"
        ce_tag = f"True(F={self.coord_freqs})" if self.use_coord_embed else "False"
        gp_tag = f"True(k={self.graph_prop_layers})" if self.use_graph_prop else "False"
        da_tag = "True" if bool(getattr(configs, "use_dense_attn", False)) else "False"
        aa_tag = f"True(K={self.adaptive_adj_dim})" if self.use_adaptive_adj else "False"
        print(
            f"[RegionFormer] enc_in={self.enc_in} | seq_len={configs.seq_len} | "
            f"pred_len={self.pred_len} | r={self.r} | d={self.d_model} | "
            f"L={self.e_layers} | scale_mode={self.scale_mode} | head={head_tag} | "
            f"E1(anchor)={self.use_distance_anchor} | E2(lambda)={self.laplacian_lambda} | "
            f"E2(active)={self.use_laplacian_smooth} | ablate_step={ablate} | "
            f"coord_embed={ce_tag} | graph_prop={gp_tag} | dense_attn={da_tag} | adaptive_adj={aa_tag}"
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.use_revin:
            x_norm = self.revin(x_enc, "norm")
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_norm = x_enc - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev

        enc_in = self.enc_embedding(x_norm)        # (B, N, D)
        if self.use_coord_embed and self.coord_emb.numel() > 0:
            # Project Fourier features (N, 4F) -> (N, D) and broadcast over batch.
            ce = self.coord_proj(self.coord_emb).unsqueeze(0)   # (1, N, D)
            enc_in = enc_in + ce
        if self.use_graph_prop and self.A_hat.numel() > 0:
            # E4: gated graph residual = γ * (A^k @ enc_in). γ initialized 0.
            # If --use_adaptive_adj, blend learned residual into A.
            if self.use_adaptive_adj:
                # A_learned = softmax(E1 E2^T / sqrt(K)) — row-stochastic, sums to 1
                logits = torch.matmul(self.adj_emb1.weight, self.adj_emb2.weight.t()) / (self.adaptive_adj_dim ** 0.5)
                A_learned = F.softmax(logits, dim=-1)
                # Blend: A_used = A_hat + alpha * (A_learned - A_hat). At init (alpha=0): A_hat.
                A_used = self.A_hat + self.adaptive_adj_gate * (A_learned - self.A_hat)
            else:
                A_used = self.A_hat
            h = enc_in
            for _ in range(self.graph_prop_layers):
                h = torch.matmul(A_used, h)
            enc_in = enc_in + self.graph_gate * h
        enc_out = self.encoder(enc_in) + enc_in    # residual connection across the stack

        if self.indep_head:
            feats = enc_out.unbind(dim=1)          # tuple of (B, D) per channel
            outs = [self.output[i](feats[i]) for i in range(self.enc_in)]
            pred = torch.stack(outs, dim=1)        # (B, N, pred_len)
        else:
            pred = self.output(enc_out)            # (B, N, pred_len)
        pred = pred.permute(0, 2, 1)               # (B, pred_len, N)

        if self.use_revin:
            out = self.revin(pred, "denorm")
        else:
            out = pred * stdev + means

        if self.use_laplacian_smooth and self.training:
            aux = self._laplacian_loss()
            if aux is not None:
                return out, aux
        return out

    def set_spatial_metadata(self, adj, coord=None, admin_groups=None):
        """Build the symmetric edge list (E2 Laplacian) and seed E1 anchors
        per layer via K-means (with optional E3 admin-seeded init).

        Mirrors SpatialLCA.set_spatial_metadata; routes anchor init through
        each layer's embedded LCA assignment module.
        """
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().cpu().numpy()
        adj = np.asarray(adj, dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj must be square 2D, got {adj.shape}")
        A = (adj + adj.T) * 0.5
        np.fill_diagonal(A, 0.0)
        upper = np.triu(A, k=1)
        src, dst = np.nonzero(upper)
        if src.size > 0:
            ei = torch.from_numpy(np.stack([src, dst], axis=0).astype(np.int64))
            ew = torch.from_numpy(upper[src, dst].astype(np.float32))
            device = self.edge_index.device
            self.edge_index = ei.to(device)
            self.edge_weight = ew.to(device)
            print(f"[RegionFormer] spatial metadata: N={adj.shape[0]}, E={src.size}")
        else:
            print("[RegionFormer] set_spatial_metadata: adjacency has no edges; smoothness disabled")

        # E4 graph propagation: build symmetric normalised adjacency
        # A_hat = D^-0.5 (A+I) D^-0.5, used by _graph_propagate in forward.
        if self.use_graph_prop:
            A_self = A + np.eye(A.shape[0], dtype=np.float32)
            d = A_self.sum(axis=1)
            d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0).astype(np.float32)
            A_norm = (A_self * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
            device = self.A_hat.device if self.A_hat.numel() else None
            self.A_hat = torch.from_numpy(A_norm)
            if device is not None:
                self.A_hat = self.A_hat.to(device)
            print(f"[RegionFormer] E4 A_hat: shape={A_norm.shape}, "
                  f"density={(A_self > 0).sum() / A_self.size:.4f}")

        if coord is not None and (self.use_distance_anchor or self.use_coord_embed):
            coord = np.asarray(coord, dtype=np.float32)
            if coord.ndim != 2 or coord.shape[1] != 2:
                raise ValueError(f"coord must be (N, 2) lat/lng, got {coord.shape}")
            cmin = coord.min(axis=0, keepdims=True)
            cmax = coord.max(axis=0, keepdims=True)
            scale = cmax - cmin
            scale[scale < 1e-9] = 1.0
            coord_norm = ((coord - cmin) / scale).astype(np.float32)
            coord_t = torch.from_numpy(coord_norm)
            if self.use_distance_anchor:
                for layer in self.encoder.layers:
                    layer.attention.assign.set_coord_and_init(coord_t, admin_groups=admin_groups)
                tag = " (E3 admin-seeded)" if admin_groups is not None else ""
                print(f"[RegionFormer] E1 anchors initialised from K-means (N={coord.shape[0]}){tag}")
            if self.use_coord_embed:
                # NeRF-style Fourier features: for each coord c in [0, 1]^2,
                #   f(c) = [sin(2^k * pi * c), cos(2^k * pi * c)] for k in 0..F-1
                # Stored as a buffer (constants); projected through learnable
                # self.coord_proj inside forward() so gradients flow.
                F = self.coord_freqs
                freqs = (2.0 ** torch.arange(F, dtype=torch.float32)) * math.pi  # (F,)
                proj = coord_t.unsqueeze(-1) * freqs                              # (N, 2, F)
                fourier = torch.cat([proj.sin(), proj.cos()], dim=-1)             # (N, 2, 2F)
                fourier = fourier.flatten(start_dim=1).contiguous()               # (N, 4F)
                # Store on same device as existing buffer
                self.coord_emb = fourier.to(self.coord_emb.device)
                print(f"[RegionFormer] coord Fourier features built: shape={tuple(fourier.shape)}")

    def _laplacian_loss(self):
        """E2 regulariser:  Σ_{layer, edges} edge_weight · ||W[..,i] - W[..,j]||²,
        averaged over batch. Activated when use_laplacian_smooth=True."""
        if self.edge_index.numel() == 0:
            return None
        ws = []
        for layer in self.encoder.layers:
            W = getattr(layer.attention, "last_W", None)
            if W is not None:
                ws.append(W)
        if not ws:
            return None
        src, dst = self.edge_index[0], self.edge_index[1]
        total = 0.0
        for W in ws:
            diff = W.index_select(-1, src) - W.index_select(-1, dst)  # (B, H, r, E)
            sq = diff.pow(2).sum(dim=2)                                # (B, H, E)
            weighted = sq * self.edge_weight
            total = total + weighted.sum() / W.shape[0]
        return self.laplacian_lambda * total
