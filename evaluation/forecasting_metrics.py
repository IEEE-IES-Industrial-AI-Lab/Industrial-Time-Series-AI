"""
Forecasting evaluation metrics for industrial time-series models.

Implements RMSE, MAE, MAPE, and SMAPE — standard metrics used in IEEE IES
and NeurIPS/ICLR time-series benchmarks.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ForecastingMetrics:
    """Bundles all forecasting metrics into a single object."""

    rmse: float
    mae: float
    mape: float
    smape: float
    model_name: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    pred_len: Optional[int] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name or "unknown",
            "dataset": self.dataset_name or "unknown",
            "pred_len": self.pred_len or -1,
            "RMSE": round(self.rmse, 4),
            "MAE": round(self.mae, 4),
            "MAPE": round(self.mape, 4),
            "SMAPE": round(self.smape, 4),
        }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%).
    
    Clips near-zero actuals to avoid division instability.
    Returns a percentage value (e.g. 5.2 means 5.2%).
    """
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    More robust than MAPE when actuals are close to zero.
    Returns a percentage value (e.g. 5.2 means 5.2%).
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def compute_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    pred_len: Optional[int] = None,
) -> ForecastingMetrics:
    """Compute all forecasting metrics from true and predicted arrays.

    Args:
        y_true: Ground truth values, shape (N,) or (N, horizon) or (N, horizon, features).
        y_pred: Predicted values, same shape as y_true.
        model_name: Optional model identifier for reporting.
        dataset_name: Optional dataset identifier for reporting.
        pred_len: Optional prediction horizon length.

    Returns:
        ForecastingMetrics dataclass.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    return ForecastingMetrics(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mape=mape(y_true, y_pred),
        smape=smape(y_true, y_pred),
        model_name=model_name,
        dataset_name=dataset_name,
        pred_len=pred_len,
    )


def print_forecasting_metrics(metrics: ForecastingMetrics) -> None:
    """Print a formatted metrics table to stdout."""
    name = metrics.model_name or "Model"
    dataset = metrics.dataset_name or "Dataset"
    header = f"  {name} on {dataset}"
    if metrics.pred_len:
        header += f" (pred_len={metrics.pred_len})"

    print(header)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'RMSE':<10} {metrics.rmse:.4f}")
    print(f"  {'MAE':<10} {metrics.mae:.4f}")
    print(f"  {'MAPE':<10} {metrics.mape:.2f}%")
    print(f"  {'SMAPE':<10} {metrics.smape:.2f}%")
    print()


def print_metrics_comparison_table(results: list[ForecastingMetrics]) -> None:
    """Print a markdown-style comparison table for multiple models/datasets.

    Args:
        results: List of ForecastingMetrics objects.
    """
    print(f"\n{'Model':<20} {'Dataset':<15} {'pred_len':>8} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8} {'SMAPE%':>8}")
    print("-" * 80)
    for m in results:
        model = (m.model_name or "unknown")[:19]
        dataset = (m.dataset_name or "unknown")[:14]
        pred = str(m.pred_len) if m.pred_len else "-"
        print(
            f"{model:<20} {dataset:<15} {pred:>8} {m.rmse:>8.4f} {m.mae:>8.4f}"
            f" {m.mape:>7.2f}% {m.smape:>7.2f}%"
        )
    print()
