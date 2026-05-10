"""CycleNet — Residual Cycle Forecasting (RCF) for time series.

Lin et al., NeurIPS 2024. Original repo: github.com/acat-scut/CycleNet

Core idea:
  1. Learn a fixed-length cyclic pattern Q ∈ R^{cycle_len × N} as a parameter.
  2. Subtract the appropriate cycle slice from input → residual signal.
  3. Forecast the residual via a Linear (or shallow MLP).
  4. Add the cycle slice for the prediction window back to the residual forecast.

No spatial information (graph/coords) used — pure-periodicity baseline.

For our v3 (CalGeo-Bench), this is a strong 2024-2025 non-spatial baseline that
naturally exploits the daily / weekly / yearly cycles in Solar, Weather, and
Mobility data — directly testing whether spatial inductive biases (RegionFormer's
E1+E2+E3) add value beyond pure periodic modelling.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from core.registry import register_model


class RecurrentCycle(nn.Module):
    """Learnable cycle pattern of length `cycle_len` over `n_channels`.

    For an input timestamp t (in absolute index), the cycle value is
    Q[t mod cycle_len, :].
    """

    def __init__(self, cycle_len: int, n_channels: int):
        super().__init__()
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(cycle_len, n_channels))

    def forward(self, index: torch.Tensor, length: int):
        """Given a 1-D start-index tensor (B,) and a sequence length, return
        (B, length, N) cycle slices wrapped modulo cycle_len."""
        # index: (B,) → arange offsets → (B, length) of t indices
        offsets = torch.arange(length, device=index.device).unsqueeze(0)  # (1, L)
        ts = (index.unsqueeze(1) + offsets) % self.cycle_len               # (B, L)
        return self.data[ts]                                                # (B, L, N)


@register_model(
    "CycleNet",
    paper="CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns",
    year=2024,
)
class Model(nn.Module):
    """CycleNet top-level model.

    Config knobs:
      - cycle_len: length of the periodic cycle (e.g. 24 for hourly-daily,
        7 for weekly, 144 for 10-min × daily). Default 24; override per-dataset.
      - model_type: 'linear' (default) or 'mlp' for the residual predictor.
      - use_revin: per-channel reversible instance norm (default True).
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = int(getattr(configs, "cycle_len", 24))
        self.model_type = getattr(configs, "cyclenet_type", "linear")  # linear | mlp
        self.use_revin = bool(getattr(configs, "use_revin", True))

        self.cycle = RecurrentCycle(self.cycle_len, self.enc_in)

        if self.model_type == "mlp":
            d_model = getattr(configs, "d_model", 512)
            self.predictor = nn.Sequential(
                nn.Linear(self.seq_len, d_model),
                nn.ReLU(),
                nn.Linear(d_model, self.pred_len),
            )
        else:
            self.predictor = nn.Linear(self.seq_len, self.pred_len)

        print(
            f"[CycleNet] enc_in={self.enc_in} | seq_len={self.seq_len} | "
            f"pred_len={self.pred_len} | cycle_len={self.cycle_len} | "
            f"type={self.model_type} | use_revin={self.use_revin}"
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                cycle_index=None):
        """x_enc: (B, T, N). Returns (B, T', N).

        cycle_index is the absolute time index of the FIRST input timestep
        (mod cycle_len). If not provided (typical training/eval where the
        framework does not track absolute time), default to 0 — equivalent to
        the cycle being aligned to the start of every window. The model is
        still expressive because the cycle parameter is jointly learned with
        the predictor.
        """
        B, T, N = x_enc.shape
        device = x_enc.device

        # RevIN per-channel (subtract mean, divide std)
        if self.use_revin:
            means = x_enc.mean(1, keepdim=True).detach()           # (B, 1, N)
            x_centered = x_enc - means
            stdev = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_centered / stdev
        else:
            x_norm = x_enc

        # Cycle index per batch element. Without external info, use 0.
        if cycle_index is None:
            cycle_index = torch.zeros(B, dtype=torch.long, device=device)
        else:
            cycle_index = cycle_index.to(device).long()

        # Subtract cycle from input
        cycle_in = self.cycle(cycle_index, T)                       # (B, T, N)
        residual = x_norm - cycle_in                                # (B, T, N)

        # Forecast residual via Linear over time axis (per-channel)
        # residual: (B, T, N) → permute → (B, N, T) → predictor → (B, N, T')
        pred_residual = self.predictor(residual.transpose(1, 2)).transpose(1, 2)
        # pred_residual: (B, T', N)

        # Add cycle for prediction window (shifted by T)
        pred_index = (cycle_index + T) % self.cycle_len             # (B,)
        cycle_pred = self.cycle(pred_index, self.pred_len)          # (B, T', N)
        pred = pred_residual + cycle_pred                            # (B, T', N)

        # RevIN inverse
        if self.use_revin:
            pred = pred * stdev + means

        return pred
