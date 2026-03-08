"""
DLinear: Decomposition Linear Model for Time-Series Forecasting.

Reference:
    Zeng et al. "Are Transformers Effective for Time Series Forecasting?"
    AAAI 2023. https://arxiv.org/abs/2205.13504

Key insight: decompose the input into trend + residual components, then apply
separate linear projections to each. Despite its simplicity, DLinear is
competitive with or superior to Transformer-based models on several ETT
benchmarks — making it an essential baseline for any serious comparison.

Architecture:
    Input (B, seq_len, C)
        ├─ MovingAvg(kernel)  →  trend   (B, seq_len, C)
        └─ Input - trend      →  residual (B, seq_len, C)

    Linear_trend    (seq_len → pred_len, per-channel)
    Linear_residual (seq_len → pred_len, per-channel)

    Output = trend_proj + residual_proj   →  (B, pred_len, C)
"""

import torch
import torch.nn as nn


class _MovingAvg(nn.Module):
    """Causal moving average for trend extraction."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # Symmetric padding to preserve sequence length
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)  →  pool expects (B, C, L)
        x = x.permute(0, 2, 1)
        # Pad both ends to maintain length
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x = nn.functional.pad(x, (pad_left, pad_right), mode="replicate")
        x = self.avg(x)
        return x.permute(0, 2, 1)  # (B, L, C)


class _SeriesDecomposition(nn.Module):
    """Decompose a time series into (residual, trend)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        residual = x - trend
        return residual, trend


class DLinear(nn.Module):
    """Channel-independent DLinear forecaster.

    Args:
        seq_len:     Input sequence length (lookback window).
        pred_len:    Forecast horizon.
        num_features: Number of input channels (sensors).
        kernel_size: Moving average kernel for trend extraction (default 25).
        individual:  If True, each channel gets its own linear weights.
                     If False, all channels share weights (faster, less memory).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        kernel_size: int = 25,
        individual: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.individual = individual

        self.decomposition = _SeriesDecomposition(kernel_size)

        if individual:
            self.linear_trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(num_features)]
            )
            self.linear_residual = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(num_features)]
            )
        else:
            self.linear_trend = nn.Linear(seq_len, pred_len)
            self.linear_residual = nn.Linear(seq_len, pred_len)

        self._init_weights()

    def _init_weights(self):
        if self.individual:
            for lin in self.linear_trend:
                nn.init.xavier_uniform_(lin.weight)
            for lin in self.linear_residual:
                nn.init.xavier_uniform_(lin.weight)
        else:
            nn.init.xavier_uniform_(self.linear_trend.weight)
            nn.init.xavier_uniform_(self.linear_residual.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, num_features)
        Returns:
            out: (B, pred_len, num_features)
        """
        residual, trend = self.decomposition(x)  # each: (B, L, C)

        if self.individual:
            # Process each channel independently
            trend_out = torch.stack(
                [self.linear_trend[c](trend[:, :, c]) for c in range(self.num_features)],
                dim=-1,
            )  # (B, pred_len, C)
            residual_out = torch.stack(
                [self.linear_residual[c](residual[:, :, c]) for c in range(self.num_features)],
                dim=-1,
            )
        else:
            # Shared weights across channels: transpose to (B, C, L), apply, transpose back
            trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            residual_out = self.linear_residual(residual.permute(0, 2, 1)).permute(0, 2, 1)

        return trend_out + residual_out  # (B, pred_len, C)
