import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from core.registry import register_model


def compute_r_svd(data: torch.Tensor, epsilon: float = 0.05) -> int:
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D tensor (T, N), got {tuple(data.shape)}")

    Xc = data - data.mean(dim=0, keepdim=True)
    k = min(Xc.shape[0], Xc.shape[1], 512)
    try:
        _, S, _ = torch.svd_lowrank(Xc, q=k)
    except Exception:
        _, S, _ = torch.svd(Xc, some=True)
        S = S[:k]

    energy = S ** 2
    total = energy.sum()
    if total < 1e-12:
        print("[compute_r_svd] WARNING: near-zero total energy, returning r*=1")
        return 1

    cumsum = energy.cumsum(dim=0)
    threshold = (1.0 - epsilon) * total
    r_star = int((cumsum < threshold).sum().item()) + 1
    r_star = max(1, min(r_star, k))

    retained = (cumsum[r_star - 1] / total).item()
    print(f"[compute_r_svd] N={Xc.shape[1]}, T={Xc.shape[0]}, "
          f"eps={epsilon:.3f} -> r*={r_star} (energy retained={retained:.4f})")
    return r_star

class SeriesDecomp(nn.Module):
    """Moving-average trend / seasonal decomposition (DLinear style)."""

    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x):
        # x: (B, L, N)
        xp = x.permute(0, 2, 1)
        xp = F.pad(xp, (self.pad, self.kernel_size - 1 - self.pad), mode="replicate")
        trend = F.avg_pool1d(xp, kernel_size=self.kernel_size, stride=1).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DishTS(nn.Module):
    """Dual-conet Dish-TS (distributional shift). Learns per-sample shift/scale
    conditioned on a compressed view of the lookback window, on top of a prior
    (z-score) normalization. Reference: Fan et al., AAAI 2023."""

    def __init__(self, num_features, seq_len, hidden_dim=8):
        super().__init__()
        self.eps = 1e-5
        self.phi_mean = nn.Linear(seq_len, hidden_dim, bias=False)
        self.phi_std = nn.Linear(seq_len, hidden_dim, bias=False)
        self.gate_mean = nn.Linear(hidden_dim, 1)
        self.gate_std = nn.Linear(hidden_dim, 1)

    def forward(self, x, mode):
        # x: (B, L, N)
        if mode == "norm":
            xp = x.permute(0, 2, 1)  # (B, N, L)
            z_mean = self.gate_mean(self.phi_mean(xp)).squeeze(-1)  # (B, N)
            z_std = self.gate_std(self.phi_std(xp)).squeeze(-1)  # (B, N)
            self.mu = (xp.mean(dim=-1) + z_mean).unsqueeze(1)  # (B, 1, N)
            self.sigma = torch.sqrt(xp.var(dim=-1, unbiased=False) + F.softplus(z_std) + self.eps).unsqueeze(1)
            return (x - self.mu) / self.sigma
        return x * self.sigma + self.mu


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, std_min=1e-3):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.std_min = std_min
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).clamp(min=self.std_min).detach()
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * self.std + self.mean


class MLPMixerBlock(nn.Module):
    """TSMixer-style time+channel mixing with residuals. Input/output: (B, L, N)."""
    def __init__(self, seq_len, enc_in, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(enc_in)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
        )
        self.norm2 = nn.LayerNorm(enc_in)
        self.chan_mlp = nn.Sequential(
            nn.Linear(enc_in, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, enc_in),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, N)
        y = self.norm1(x)
        y = self.time_mlp(y.transpose(-1, -2)).transpose(-1, -2)  # MLP over L
        x = x + self.dropout(y)
        y = self.norm2(x)
        y = self.chan_mlp(y)  # MLP over N
        return x + self.dropout(y)


class MLPMixerBranch(nn.Module):
    """Stacked MLPMixerBlocks + linear projection L->pred_len. Per-channel independent along N."""
    def __init__(self, seq_len, pred_len, enc_in, hidden_dim, n_blocks=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLPMixerBlock(seq_len, enc_in, hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, L, N)
        for b in self.blocks:
            x = b(x)
        # project along time: (B, L, N) -> (B, pred_len, N)
        return self.proj(x.transpose(-1, -2)).transpose(-1, -2)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout=0.1):
        super().__init__()
        self.hidden = nn.Linear(in_dim, h_dim)
        self.output = nn.Linear(h_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.output(self.act(self.hidden(x)))) + self.residual(x)


class InvertedPatchEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, patch_size=14, patch_stride=7, dropout=0.1, use_time_enc=False, keep_patches=False, grad_ckpt=False, channel_chunk=0):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.use_time_enc = use_time_enc
        self.keep_patches = keep_patches
        self.grad_ckpt = grad_ckpt
        self.channel_chunk = channel_chunk

        self.patch_num = (seq_len - patch_size) // patch_stride + 1
        in_dim = patch_size * 2 if use_time_enc else patch_size

        self.value_embedding = ResidualBlock(in_dim, d_model, d_model, dropout)
        if not keep_patches:
            self.out = nn.Linear(self.patch_num * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_time_enc:
            t = torch.arange(-seq_len, 0).float() / seq_len
            self.register_buffer('time_enc', t)

    def forward(self, x):
        B, L, N = x.shape
        x = x.permute(0, 2, 1).contiguous()  # (B, N, L)

        def _embed(x_chunk):
            # x_chunk: (B, n, L)
            if self.use_time_enc:
                n = x_chunk.shape[1]
                t = self.time_enc.unsqueeze(0).unsqueeze(0).expand(B, n, L)
                xt = torch.stack([x_chunk, t], dim=-1).reshape(B, n, L * 2)
                xt = xt.unfold(-1, self.patch_size * 2, self.patch_stride * 2).contiguous()
            else:
                xt = x_chunk.unfold(-1, self.patch_size, self.patch_stride).contiguous()
            if self.grad_ckpt and self.training:
                return torch.utils.checkpoint.checkpoint(self.value_embedding, xt, use_reentrant=False)
            return self.value_embedding(xt)

        if self.channel_chunk and self.channel_chunk < N:
            outs = []
            for i in range(0, N, self.channel_chunk):
                outs.append(_embed(x[:, i:i + self.channel_chunk, :]))
            x = torch.cat(outs, dim=1)  # (B, N, patch_num, d_model)
        else:
            x = _embed(x)

        if self.keep_patches:
            return self.dropout(x)
        x = x.reshape(B, N, -1).contiguous()
        return self.dropout(self.out(x))

class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, **kwargs):
        B, L, H, E = queries.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), A


class LatentCentroidAttention(nn.Module):
    def __init__(self, d_model, n_heads, r, attention_dropout=0.1, gumbel_alpha=0.15,
                 use_distance_anchor=False, distance_alpha=1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.r = r
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads
        self.gumbel_alpha = gumbel_alpha

        self.inner_attention = FullAttention(attention_dropout)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

        self.assign_net = nn.Linear(d_model, n_heads * r)
        self.log_tau = nn.Parameter(torch.zeros(1))

        # E1: distance-decay anchored centroids. Anchors are 2D positions per
        # (head, centroid); soft assignment biased by exp(−α·d / h) where d is
        # node→anchor distance. Falls back to assign_net when use_distance_anchor=False.
        self.use_distance_anchor = use_distance_anchor
        if use_distance_anchor:
            self.anchor = nn.Parameter(torch.zeros(n_heads, r, 2))
            self.log_h = nn.Parameter(torch.zeros(n_heads))  # exp(0)=1.0
            self.distance_alpha = nn.Parameter(torch.tensor(float(distance_alpha)))
            self.register_buffer("coord", torch.empty(0, 2), persistent=False)

    def set_coord_and_init(self, coord_norm: torch.Tensor, admin_groups=None):
        """coord_norm: (N, 2) normalised lat/lng in [0, 1]. Init anchors via K-means.

        E3-lite: if `admin_groups` is given (shape (N,) of admin labels — e.g. county
        FIPS), use the per-group centroids as K-means initialisation seeds. This
        embeds geographic admin scale into anchor initialisation without changing
        the model architecture; centroids fine-tune to admin geography first.
        """
        if not self.use_distance_anchor:
            return
        self.coord = coord_norm.to(self.coord.device if self.coord.numel() else coord_norm.device)
        try:
            from sklearn.cluster import KMeans
            X = coord_norm.cpu().numpy()
            if admin_groups is not None:
                # Compute per-group mean coord to seed K-means
                groups = np.asarray(admin_groups)
                uniq = np.unique(groups)
                seeds = np.stack([X[groups == g].mean(axis=0) for g in uniq], axis=0)
                # If r_centroids < n_groups, drop tail; if r > n_groups, pad with random points
                if self.r <= len(seeds):
                    seeds = seeds[:self.r]
                else:
                    rng = np.random.default_rng(2024)
                    extra = X[rng.integers(0, len(X), size=self.r - len(seeds))]
                    seeds = np.concatenate([seeds, extra], axis=0)
                km = KMeans(n_clusters=self.r, init=seeds.astype(X.dtype), n_init=1, random_state=2024).fit(X)
                print(f"[LatentCentroidAttention] anchors seeded from {len(uniq)} admin groups (E3-lite init)")
            else:
                km = KMeans(n_clusters=self.r, n_init=4, random_state=2024).fit(X)
            centers = torch.from_numpy(km.cluster_centers_.astype("float32"))
            with torch.no_grad():
                self.anchor.copy_(centers.unsqueeze(0).expand(self.n_heads, -1, -1))
        except ImportError:
            print("[LatentCentroidAttention] sklearn unavailable; anchors stay at zero")

    def _assignment(self, x):
        B, N, _ = x.shape
        tau = torch.clamp(self.log_tau.exp(), min=0.1, max=10.0)

        logits = self.assign_net(x) # (B, N, H*r)
        logits = logits.view(B, N, self.n_heads, self.r)
        logits = logits.permute(0,2,3,1) # (B, H, r, N)

        if self.use_distance_anchor and self.coord.numel() > 0:
            # coord: (N, 2), anchor: (H, r, 2). Compute pairwise L2 distance.
            # clamp_min(eps) avoids sqrt(0) backward = inf when an anchor lands
            # exactly on a sensor coord (common after K-means init seeded by sensors).
            diff = self.coord.unsqueeze(0).unsqueeze(0) - self.anchor.unsqueeze(2)  # (H, r, N, 2)
            d = diff.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()  # (H, r, N)
            h = self.log_h.exp().clamp(min=0.01, max=10.0).view(-1, 1, 1)  # (H, 1, 1)
            spatial_bias = -self.distance_alpha * d / h  # (H, r, N)
            logits = logits + spatial_bias.unsqueeze(0)  # broadcast over B

        if self.training and self.gumbel_alpha > 0.0:
            U = torch.empty(logits.shape, device=logits.device, dtype=torch.float32).uniform_().clamp_(1e-6, 1 - 1e-6)
            G = -torch.log(-torch.log(U))
            logits = logits + self.gumbel_alpha * G.to(logits.dtype)

        return F.softmax(logits / tau, dim=-1)

    def forward(self, x, attn_mask=None, **kwargs):
        B, N, D = x.shape

        q = self.query_projection(x).view(B, N, self.n_heads, self.d_keys)
        k = self.key_projection(x).view(B, N, self.n_heads, self.d_keys)
        v = self.value_projection(x).view(B, N, self.n_heads, self.d_values)

        W = self._assignment(x)
        # Stash for SpatialLCA's Laplacian smoothness regulariser. Detach=False:
        # we want the regulariser's gradients to flow into assign_net.
        self.last_W = W
        k_cent = torch.einsum("bnhd,bhrn->brhd", k, W)
        v_cent = torch.einsum("bnhd,bhrn->brhd", v, W)

        out, _ = self.inner_attention(q, k_cent, v_cent, attn_mask)
        return self.out_projection(out.reshape(B, N, D))


class VanillaAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.inner = FullAttention(attention_dropout)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None, **kwargs):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.n_heads, self.d_head)
        k = self.k(x).view(B, N, self.n_heads, self.d_head)
        v = self.v(x).view(B, N, self.n_heads, self.d_head)
        o, _ = self.inner(q, k, v, attn_mask)
        return self.out(o.reshape(B, N, D))


class LinformerAttention(nn.Module):
    """Attention with K/V projected along the token (channel) dim to rank k.
    Cost: O(N * k * d) instead of O(N^2 * d). Assumes fixed N (enc_in)."""
    def __init__(self, d_model, n_heads, n_tokens, k=64, attention_dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = min(k, n_tokens)
        self.inner = FullAttention(attention_dropout)
        self.q = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.E = nn.Parameter(torch.empty(self.k, n_tokens))
        self.F = nn.Parameter(torch.empty(self.k, n_tokens))
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None, **kwargs):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.n_heads, self.d_head)
        k = self.k_proj(x)  # (B, N, D)
        v = self.v_proj(x)  # (B, N, D)
        # project along N -> k
        k_proj = torch.einsum("kn,bnd->bkd", self.E, k).view(B, self.k, self.n_heads, self.d_head)
        v_proj = torch.einsum("kn,bnd->bkd", self.F, v).view(B, self.k, self.n_heads, self.d_head)
        o, _ = self.inner(q, k_proj, v_proj, attn_mask)
        return self.out(o.reshape(B, N, D))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, r, d_ff=None, dropout=0.1,
                 activation='gelu', attention_dropout=0.1, gumbel_alpha=0.15,
                 attn_type='lca', n_tokens=None, linformer_k=64,
                 use_distance_anchor=False, distance_alpha=1.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        if attn_type == 'full':
            self.attention = VanillaAttention(d_model, n_heads, attention_dropout)
        elif attn_type == 'linformer':
            assert n_tokens is not None, "linformer needs n_tokens (enc_in)"
            self.attention = LinformerAttention(d_model, n_heads, n_tokens, k=linformer_k, attention_dropout=attention_dropout)
        else:
            self.attention = LatentCentroidAttention(
                d_model, n_heads, r, attention_dropout, gumbel_alpha=gumbel_alpha,
                use_distance_anchor=use_distance_anchor, distance_alpha=distance_alpha,
            )
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

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


@register_model("SpatialLCA", paper="ScaleX: Latent Centroid Attention for High-Dimensional Time Series", year=2026)
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.use_laplacian_smooth = getattr(configs, "use_laplacian_smooth", False)
        self.laplacian_lambda = float(getattr(configs, "laplacian_lambda", 0.01))
        # Sparse undirected edge list — populated via set_spatial_metadata().
        # Buffers so they move with .to(device); not persisted to checkpoints.
        self.register_buffer("edge_index", torch.empty(2, 0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_weight", torch.empty(0, dtype=torch.float32), persistent=False)
        # E1: anchored centroids
        self.use_distance_anchor = getattr(configs, "use_distance_anchor", False)
        self.distance_alpha = float(getattr(configs, "distance_alpha", 1.0))
        # E4: graph propagation residual (uses set_spatial_metadata's adj as a
        # normalised propagation operator, applied per-feature on encoder input
        # AND output, with a learnable scalar gate.)
        self.use_graph_prop = getattr(configs, "use_graph_prop", False)
        self.graph_prop_layers = int(getattr(configs, "graph_prop_layers", 1))
        if self.use_graph_prop:
            self.register_buffer("A_hat", torch.empty(0), persistent=False)
            # init gates at 0 so the model behaves identically to baseline at start
            # and only learns to use graph prop if it helps. STGNN-style.
            self.graph_gate_pre = nn.Parameter(torch.tensor(0.0))
            self.graph_gate_post = nn.Parameter(torch.tensor(0.0))
            self.graph_prop_position = getattr(configs, "graph_prop_position", "pre")  # pre | post | both
        # E5: parallel GCN branch fused at encoder output. Two-layer message passing
        # over road adjacency, late-fused with LCA via sigmoid gate.
        self.use_gcn_branch = getattr(configs, "use_gcn_branch", False)
        self.gcn_layers = int(getattr(configs, "gcn_layers", 2))
        if self.use_gcn_branch:
            if not self.use_graph_prop:
                # we still need A_hat; share buffer
                self.register_buffer("A_hat", torch.empty(0), persistent=False)
            self.gcn_lin = nn.ModuleList([
                nn.Linear(getattr(configs, "d_model"), getattr(configs, "d_model"))
                for _ in range(self.gcn_layers)
            ])
            self.gcn_norm = nn.LayerNorm(getattr(configs, "d_model"))
            # init gate at -3 so sigmoid(-3) ≈ 0.05 — model starts mostly LCA, learns to lean GCN.
            self.gcn_gate = nn.Parameter(torch.tensor(-3.0))
        self.r = configs.r_star
        self.dropout = configs.dropout
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.pred_len = configs.pred_len
        self.gumbel_alpha = configs.gumbel_alpha
        self.use_time_enc = configs.use_time_enc
        self.e_layers = configs.e_layers
        self.use_revin = getattr(configs, "use_revin", False)
        if self.use_revin:
            self.revin = RevIN(configs.enc_in, affine=not getattr(configs, "revin_no_affine", False))
        self.use_dishts = getattr(configs, "use_dishts", False)
        if self.use_dishts:
            self.dishts = DishTS(configs.enc_in, configs.seq_len)
        self.use_decomp = getattr(configs, "use_decomp", False)
        if self.use_decomp:
            self.decomp = SeriesDecomp(kernel_size=getattr(configs, "decomp_kernel", 25))
            self.trend_linear = nn.Linear(configs.seq_len, configs.pred_len)
        self.mlp_head = getattr(configs, "mlp_head", False)
        self.indep_head = getattr(configs, "indep_head", False)
        self.head_hidden_ratio = getattr(configs, "head_hidden_ratio", 1)
        self.subtract_last = getattr(configs, "subtract_last", False)
        self.keep_patches = getattr(configs, "keep_patches", False)
        self.enc_in = configs.enc_in
        self.patch_num = (configs.seq_len - configs.patch_size) // configs.patch_stride + 1

        self.enc_embedding = InvertedPatchEmbedding(
            seq_len=configs.seq_len,
            d_model=self.d_model,
            patch_size=configs.patch_size,
            patch_stride=configs.patch_stride,
            dropout=self.dropout,
            use_time_enc=self.use_time_enc,
            keep_patches=self.keep_patches,
            grad_ckpt=getattr(configs, "grad_ckpt", False),
            channel_chunk=getattr(configs, "channel_chunk", 0),
            )

        self.attn_type = getattr(configs, "attn", "lca")
        self.backbone = getattr(configs, "backbone", "lca")
        self.use_mixer_aux = getattr(configs, "use_mixer_aux", False) or self.backbone == "mixer"
        if self.use_mixer_aux:
            self.mixer = MLPMixerBranch(
                seq_len=configs.seq_len,
                pred_len=configs.pred_len,
                enc_in=configs.enc_in,
                hidden_dim=getattr(configs, "mixer_hidden", 256),
                n_blocks=getattr(configs, "mixer_blocks", 2),
                dropout=self.dropout,
            )
            self.mixer_gate = nn.Parameter(torch.tensor(0.0))
        self.encoder = Encoder([
            EncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                r=self.r,
                d_ff = configs.d_ff or self.d_model * 4,
                dropout=self.dropout,
                activation='gelu',
                attention_dropout=self.dropout,
                gumbel_alpha=self.gumbel_alpha,
                attn_type=self.attn_type,
                n_tokens=configs.enc_in,
                linformer_k=getattr(configs, "linformer_k", 64),
                use_distance_anchor=self.use_distance_anchor,
                distance_alpha=self.distance_alpha,
            )
            for _ in range(self.e_layers)
        ], norm_layer=nn.LayerNorm(self.d_model), grad_ckpt=getattr(configs, "grad_ckpt", False))

        h_dim = self.d_model * self.head_hidden_ratio
        head_in_dim = self.patch_num * self.d_model if self.keep_patches else self.d_model

        def _build_head():
            if self.mlp_head:
                return nn.Sequential(
                    nn.Linear(head_in_dim, h_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(h_dim, self.pred_len),
                )
            return nn.Linear(head_in_dim, self.pred_len)

        if self.indep_head:
            self.output = nn.ModuleList([_build_head() for _ in range(self.enc_in)])
        else:
            self.output = _build_head()

        print(f"[SpatialLCA] enc_in={configs.enc_in} | seq_len={configs.seq_len} | "
              f"pred_len={configs.pred_len} | r={configs.r_star} | d={self.d_model} | "
              f"alpha={configs.gumbel_alpha}")

    def _graph_propagate(self, h, k=1):
        """Apply k iterations of A_hat @ h on the channel dim of (B, N, ...) tensor."""
        if self.A_hat.numel() == 0:
            return h
        out = h
        orig_shape = out.shape
        flat = out.reshape(orig_shape[0], orig_shape[1], -1)  # (B, N, F)
        for _ in range(k):
            flat = torch.matmul(self.A_hat, flat)  # (N, N) @ (B, N, F) -> broadcasts
        return flat.reshape(orig_shape)

    def _gcn_branch_forward(self, h):
        """E5 parallel GCN block: 2-layer GCN over A_hat. Input/output shape (B, N, d_model)."""
        if self.A_hat.numel() == 0:
            return h
        out = h
        for i, lin in enumerate(self.gcn_lin):
            # propagate then transform
            propagated = torch.matmul(self.A_hat, out)  # (N,N) @ (B,N,D) broadcasts
            out = lin(propagated)
            if i < len(self.gcn_lin) - 1:
                out = torch.nn.functional.gelu(out)
        return self.gcn_norm(out)

    def _encoder_forward(self, x):
        enc_in = self.enc_embedding(x)  # (B, N, d_model) or (B, N, patch_num, d_model) if keep_patches
        # E4 pre: graph residual on encoder input
        if self.use_graph_prop and self.A_hat.numel() > 0 and self.graph_prop_position in ("pre", "both"):
            enc_in = enc_in + self.graph_gate_pre * self._graph_propagate(enc_in, k=self.graph_prop_layers)
        if self.keep_patches:
            B, N, P, D = enc_in.shape
            enc_in_flat = enc_in.reshape(B * N, P, D)
            enc_out_flat = self.encoder(enc_in_flat)
            feat_flat = enc_out_flat + enc_in_flat  # (B*N, P, D)
            feat = feat_flat.reshape(B, N, P * D)  # flatten patches for head
        else:
            enc_out = self.encoder(enc_in)
            feat = enc_out + enc_in  # (B, N, d_model)
        # E4 post: graph residual on encoder output
        if self.use_graph_prop and self.A_hat.numel() > 0 and self.graph_prop_position in ("post", "both"):
            feat = feat + self.graph_gate_post * self._graph_propagate(feat, k=self.graph_prop_layers)
        # E5: late-fuse with parallel GCN branch (uses pre-encoder enc_in to avoid loop with feat)
        if self.use_gcn_branch and not self.keep_patches and self.A_hat.numel() > 0:
            gcn_feat = self._gcn_branch_forward(enc_in)
            gate = torch.sigmoid(self.gcn_gate)
            feat = feat + gate * gcn_feat
        if self.indep_head:
            # per-channel head: split along N, apply each head
            feats = feat.unbind(dim=1)  # tuple of (B, feat_dim) per channel
            outs = [self.output[i](feats[i]) for i in range(self.enc_in)]
            pred = torch.stack(outs, dim=1)  # (B, N, pred_len)
        else:
            pred = self.output(feat)  # (B, N, pred_len)
        return pred.permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.use_revin:
            x_norm = self.revin(x_enc, "norm")
        elif self.use_dishts:
            x_norm = self.dishts(x_enc, "norm")
        elif self.subtract_last:
            last = x_enc[:, -1:, :].detach()
            x_norm = x_enc - last
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_norm = x_enc - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev

        if self.backbone == "mixer":
            # pure mixer path: skip attention encoder entirely
            pred = self.mixer(x_norm)  # (B, pred_len, N)
        elif self.use_decomp:
            seasonal, trend = self.decomp(x_norm)
            seasonal_pred = self._encoder_forward(seasonal)
            trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
            pred = seasonal_pred + trend_pred
        else:
            pred = self._encoder_forward(x_norm)

        if self.use_mixer_aux and self.backbone != "mixer":
            mixer_pred = self.mixer(x_norm)  # (B, pred_len, N)
            pred = pred + torch.sigmoid(self.mixer_gate) * mixer_pred

        if self.use_revin:
            out = self.revin(pred, "denorm")
        elif self.use_dishts:
            out = self.dishts(pred, "denorm")
        elif self.subtract_last:
            out = pred + last
        else:
            out = pred * stdev + means

        if self.use_laplacian_smooth and self.training:
            aux = self._laplacian_loss()
            if aux is not None:
                return out, aux
        return out

    def set_spatial_metadata(self, adj, coord=None, admin_groups=None):
        """Build a symmetric undirected edge list (upper triangle) from a possibly
        asymmetric adjacency. If `coord` (lat/lng per node) is given AND
        use_distance_anchor is on, also normalise it and seed each LCA layer's
        anchor positions via K-means.

        E3-lite: if `admin_groups` (e.g. county FIPS per node) is given, K-means
        is seeded from per-group centroids — embeds admin scale into init.

        Stored as buffers so they ride along to the device."""
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
            print(f"[SpatialLCA] spatial metadata loaded: N={adj.shape[0]}, E={src.size}")
        else:
            print("[SpatialLCA] set_spatial_metadata: adjacency has no edges; smoothness disabled at runtime")

        # E4/E5: build symmetric normalised adjacency A_hat = D^-1/2 (A+I) D^-1/2.
        if self.use_graph_prop or self.use_gcn_branch:
            A_self = A + np.eye(A.shape[0], dtype=np.float32)
            d = A_self.sum(axis=1)
            d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0).astype(np.float32)
            A_norm = (A_self * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
            device = self.A_hat.device if self.A_hat.numel() else None
            self.A_hat = torch.from_numpy(A_norm)
            if device is not None:
                self.A_hat = self.A_hat.to(device)
            print(f"[SpatialLCA] E4 graph propagation: A_hat (N, N)={A_norm.shape}, "
                  f"density={(A_self > 0).sum() / A_self.size:.4f}")

        if coord is not None and self.use_distance_anchor:
            coord = np.asarray(coord, dtype=np.float32)
            if coord.ndim != 2 or coord.shape[1] != 2:
                raise ValueError(f"coord must be (N, 2) lat/lng, got {coord.shape}")
            # min-max normalize so anchors live in [0, 1]^2
            cmin = coord.min(axis=0, keepdims=True)
            cmax = coord.max(axis=0, keepdims=True)
            scale = (cmax - cmin)
            scale[scale < 1e-9] = 1.0
            coord_norm = ((coord - cmin) / scale).astype(np.float32)
            coord_t = torch.from_numpy(coord_norm)
            for layer in self.encoder.layers:
                attn = layer.attention
                if isinstance(attn, LatentCentroidAttention) and attn.use_distance_anchor:
                    attn.set_coord_and_init(coord_t, admin_groups=admin_groups)
            tag = " (E3-lite admin-seeded)" if admin_groups is not None else ""
            print(f"[SpatialLCA] E1 anchors initialised from K-means on coord (N={coord.shape[0]}){tag}")

    def _laplacian_loss(self):
        """Σ_{layer, b, h, r} edge_weight · (W[..,i] - W[..,j])^2, averaged over batch."""
        if self.edge_index.numel() == 0:
            return None
        ws = []
        for layer in self.encoder.layers:
            attn = layer.attention
            W = getattr(attn, "last_W", None)
            if W is not None and isinstance(attn, LatentCentroidAttention):
                ws.append(W)
        if not ws:
            return None
        src, dst = self.edge_index[0], self.edge_index[1]
        total = 0.0
        for W in ws:
            # W: (B, H, r, N)
            diff = W.index_select(-1, src) - W.index_select(-1, dst)  # (B, H, r, E)
            sq_per_edge = diff.pow(2).sum(dim=2)  # (B, H, E) — sum over r centroids
            weighted = sq_per_edge * self.edge_weight                  # broadcast E
            total = total + weighted.sum() / W.shape[0]                # mean over batch
        return self.laplacian_lambda * total