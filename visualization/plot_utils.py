"""
Visualization utilities for Industrial-Time-Series-AI.

Provides publication-ready plots for:
  - Multivariate time-series inspection
  - Anomaly score visualization with ground truth overlay
  - Forecast vs. actual comparison
  - Model comparison bar charts
  - Benchmark results tables (terminal and matplotlib)
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

_PALETTE = sns.color_palette("muted")


# ---------------------------------------------------------------------------
# Multivariate time-series plot
# ---------------------------------------------------------------------------

def plot_multivariate_timeseries(
    data: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Multivariate Industrial Time Series",
    max_subplots: int = 5,
    figsize_per_row: tuple = (12, 2),
) -> plt.Figure:
    """Plot a multivariate time-series with one subplot per sensor.

    Args:
        data:            Shape (seq_len, num_features).
        feature_names:   Optional list of sensor/feature names.
        title:           Figure title.
        max_subplots:    Maximum number of sensors to plot.
        figsize_per_row: (width, height) per subplot row.

    Returns:
        Matplotlib Figure.
    """
    seq_len, num_features = data.shape
    num_plots = min(num_features, max_subplots)
    width, row_height = figsize_per_row

    fig, axes = plt.subplots(num_plots, 1, figsize=(width, row_height * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    for i in range(num_plots):
        axes[i].plot(data[:, i], lw=1.2, alpha=0.85, color=_PALETTE[i % len(_PALETTE)])
        name = feature_names[i] if feature_names and i < len(feature_names) else f"Sensor {i + 1}"
        axes[i].set_ylabel(name, fontsize=9)
        axes[i].yaxis.set_label_position("right")

    axes[-1].set_xlabel("Time (steps)", fontsize=11)
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Anomaly score plot
# ---------------------------------------------------------------------------

def plot_anomaly_scores(
    true_data: np.ndarray,
    anomaly_scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    feature_idx: int = 0,
    title: str = "Anomaly Detection Results",
) -> plt.Figure:
    """Plot sensor reading alongside reconstruction error and ground truth labels.

    Args:
        true_data:       Raw sensor data, shape (T, num_features).
        anomaly_scores:  Per-timestep anomaly scores, shape (T,).
        labels:          Ground truth binary anomaly labels, shape (T,). Optional.
        threshold:       Decision threshold line on the score plot.
        feature_idx:     Which sensor channel to display in the top panel.
        title:           Figure title.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    # Top: sensor signal + true anomaly highlights
    ax1.plot(true_data[:, feature_idx], color=_PALETTE[0], lw=1.2, alpha=0.8, label="Sensor signal")
    if labels is not None:
        anom_idx = np.where(labels == 1)[0]
        if len(anom_idx):
            ax1.scatter(anom_idx, true_data[anom_idx, feature_idx], color="red", s=8, zorder=3, label="True anomalies")
    ax1.set_ylabel("Sensor value", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(title, fontsize=12)

    # Bottom: anomaly score + threshold
    ax2.fill_between(range(len(anomaly_scores)), anomaly_scores, alpha=0.3, color=_PALETTE[2])
    ax2.plot(anomaly_scores, color=_PALETTE[2], lw=1.2, label="Reconstruction error")
    if threshold is not None:
        ax2.axhline(threshold, color="red", linestyle="--", lw=1.5, label=f"Threshold={threshold:.4f}")
    ax2.set_xlabel("Time (steps)", fontsize=10)
    ax2.set_ylabel("Anomaly score", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Forecast vs. actual overlay
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual(
    targets: np.ndarray,
    predictions: np.ndarray,
    feature_idx: int = 0,
    title: str = "Forecast vs. Actual",
    pred_len: Optional[int] = None,
    n_context: int = 96,
) -> plt.Figure:
    """Overlay ground truth and model predictions for a single channel.

    Works with outputs of shape (N, pred_len, C) or (N, pred_len).
    Plots the last `n_context` target steps as context, then overlays
    the first `pred_len` predicted steps.

    Args:
        targets:     Ground truth array. Shape (N, pred_len, C) or (N, pred_len).
        predictions: Predicted array, same shape.
        feature_idx: Channel index to display if multi-feature.
        title:       Figure title.
        pred_len:    Forecast horizon (inferred from shape if not given).
        n_context:   How many true steps to show before the prediction window.

    Returns:
        Matplotlib Figure.
    """
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    # Extract single feature if multi-dimensional
    if targets.ndim == 3:
        targets = targets[:, :, feature_idx]
        predictions = predictions[:, :, feature_idx]
    if targets.ndim == 2:
        t_flat = targets.flatten()
        p_flat = predictions.flatten()
    else:
        t_flat = targets
        p_flat = predictions

    show_len = min(n_context + (pred_len or len(p_flat)), len(t_flat))

    fig, ax = plt.subplots(figsize=(13, 4))
    x_all = np.arange(show_len)
    ax.plot(x_all, t_flat[:show_len], label="Ground truth", color=_PALETTE[0], lw=1.5)
    ax.plot(x_all, p_flat[:show_len], label="Prediction", color=_PALETTE[1], lw=1.5, linestyle="--")

    if pred_len and n_context < show_len:
        ax.axvline(n_context, color="gray", linestyle=":", lw=1.2, label="Forecast start")

    ax.set_xlabel("Time (steps)", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison_bar(
    model_names: Sequence[str],
    metric_values: Sequence[float],
    metric_name: str = "RMSE",
    dataset_name: str = "",
    lower_is_better: bool = True,
) -> plt.Figure:
    """Bar chart comparing multiple models on a single metric.

    Args:
        model_names:    List of model identifiers.
        metric_values:  Corresponding metric values.
        metric_name:    Name of the metric (e.g. "RMSE", "F1").
        dataset_name:   Dataset label for the title.
        lower_is_better: If True, the best bar is highlighted in green; else orange.

    Returns:
        Matplotlib Figure.
    """
    n = len(model_names)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(n)]
    best_idx = int(np.argmin(metric_values) if lower_is_better else np.argmax(metric_values))
    colors[best_idx] = "#2ecc71" if lower_is_better else "#e67e22"

    fig, ax = plt.subplots(figsize=(max(7, n * 1.4), 4))
    bars = ax.bar(model_names, metric_values, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate bars with values
    for bar, val in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005 * max(metric_values),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(metric_name, fontsize=11)
    suffix = " (↓ lower is better)" if lower_is_better else " (↑ higher is better)"
    title = f"Model Comparison — {metric_name}{suffix}"
    if dataset_name:
        title += f"\nDataset: {dataset_name}"
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(metric_values) * 1.15)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Benchmark results table (matplotlib)
# ---------------------------------------------------------------------------

def plot_benchmark_table(
    rows: list[dict],
    columns: Optional[list[str]] = None,
    title: str = "Benchmark Results",
) -> plt.Figure:
    """Render benchmark results as a styled table figure.

    Args:
        rows:    List of dicts, each representing one model/dataset run.
                 Keys must match `columns`.
        columns: Ordered list of column keys to display. If None, infer from rows[0].
        title:   Figure title.

    Returns:
        Matplotlib Figure.
    """
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No results to display.", ha="center", va="center")
        return fig

    if columns is None:
        columns = list(rows[0].keys())

    cell_data = [[str(row.get(col, "")) for col in columns] for row in rows]

    fig_h = max(3, 0.5 + 0.35 * (len(rows) + 1))
    fig_w = max(8, 1.4 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header row
    for col_idx in range(len(columns)):
        table[(0, col_idx)].set_facecolor("#2c3e50")
        table[(0, col_idx)].set_text_props(color="white", fontweight="bold")

    # Alternating row shading
    for row_idx in range(1, len(rows) + 1):
        color = "#f0f4f8" if row_idx % 2 == 0 else "white"
        for col_idx in range(len(columns)):
            table[(row_idx, col_idx)].set_facecolor(color)

    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Terminal table printer
# ---------------------------------------------------------------------------

def print_benchmark_table(rows: list[dict], columns: Optional[list[str]] = None) -> None:
    """Print a formatted markdown-style benchmark table to stdout.

    Args:
        rows:    List of result dicts.
        columns: Ordered list of keys to display. If None, infer from rows[0].
    """
    if not rows:
        print("No results.")
        return
    if columns is None:
        columns = list(rows[0].keys())

    col_widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns]

    header = " | ".join(c.ljust(w) for c, w in zip(columns, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    print(header)
    print(separator)
    for row in rows:
        line = " | ".join(str(row.get(c, "")).ljust(w) for c, w in zip(columns, col_widths))
        print(line)
    print()
