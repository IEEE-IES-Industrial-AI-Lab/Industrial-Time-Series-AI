"""
Anomaly detection evaluation metrics for industrial time-series.

Implements standard metrics used in IEEE IES anomaly detection papers:
  - ROC-AUC, F1, Precision, Recall
  - Point-Adjust (PA) — the de-facto standard in ICS/SCADA anomaly benchmarks
  - Best-threshold search over the anomaly score distribution
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)


@dataclass
class AnomalyMetrics:
    """Bundles all anomaly detection metrics into a single object."""

    roc_auc: float
    f1: float
    precision: float
    recall: float
    f1_pa: float
    precision_pa: float
    recall_pa: float
    best_threshold: float
    model_name: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name or "unknown",
            "dataset": self.dataset_name or "unknown",
            "ROC-AUC": round(self.roc_auc, 4),
            "F1": round(self.f1, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1-PA": round(self.f1_pa, 4),
            "Precision-PA": round(self.precision_pa, 4),
            "Recall-PA": round(self.recall_pa, 4),
            "Threshold": round(self.best_threshold, 6),
        }


def point_adjust(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Apply the Point-Adjust (PA) strategy to binary predictions.

    If any point within a true anomaly segment is detected, the entire segment
    is credited as detected. This is the standard evaluation protocol used in
    THOC, TranAD, AnomalyTransformer, and most IES anomaly papers.

    Args:
        y_true: Ground truth binary labels (0 normal, 1 anomaly), shape (N,).
        y_pred: Binary predictions, shape (N,).

    Returns:
        Adjusted binary predictions, shape (N,).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int).copy()

    # Find contiguous anomaly segments in ground truth
    in_segment = False
    seg_start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_segment:
            in_segment = True
            seg_start = i
        elif y_true[i] == 0 and in_segment:
            in_segment = False
            seg_end = i
            # If any prediction in the segment is positive, mark all positive
            if y_pred[seg_start:seg_end].any():
                y_pred[seg_start:seg_end] = 1
    # Handle segment ending at the last index
    if in_segment:
        if y_pred[seg_start:].any():
            y_pred[seg_start:] = 1

    return y_pred


def best_threshold_search(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find the threshold that maximises F1 (or another metric) on the score distribution.

    Uses the precision-recall curve breakpoints, which avoids brute-force sweeping.

    Args:
        y_true: Ground truth binary labels, shape (N,).
        scores: Anomaly scores (higher = more anomalous), shape (N,).
        metric: Optimisation target — only "f1" is currently supported.

    Returns:
        (best_threshold, best_score) tuple.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns n+1 precision/recall values for n thresholds
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)

    if len(f1_scores) == 0:
        return float(np.median(scores)), 0.0

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def compute_anomaly_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> AnomalyMetrics:
    """Compute all anomaly detection metrics.

    Args:
        y_true: Ground truth binary labels (0/1), shape (N,).
        scores: Continuous anomaly scores (higher = more anomalous), shape (N,).
        threshold: Decision threshold. If None, best threshold from F1 search is used.
        model_name: Optional model identifier for reporting.
        dataset_name: Optional dataset identifier for reporting.

    Returns:
        AnomalyMetrics dataclass.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")

    if threshold is None:
        threshold, _ = best_threshold_search(y_true, scores)

    y_pred = (scores >= threshold).astype(int)
    y_pred_pa = point_adjust(y_true, y_pred)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    f1_pa = float(f1_score(y_true, y_pred_pa, zero_division=0))
    prec_pa = float(precision_score(y_true, y_pred_pa, zero_division=0))
    rec_pa = float(recall_score(y_true, y_pred_pa, zero_division=0))

    return AnomalyMetrics(
        roc_auc=auc,
        f1=f1,
        precision=prec,
        recall=rec,
        f1_pa=f1_pa,
        precision_pa=prec_pa,
        recall_pa=rec_pa,
        best_threshold=threshold,
        model_name=model_name,
        dataset_name=dataset_name,
    )


def print_anomaly_metrics(metrics: AnomalyMetrics) -> None:
    """Print a formatted anomaly detection metrics table."""
    name = metrics.model_name or "Model"
    dataset = metrics.dataset_name or "Dataset"
    header = f"  {name} on {dataset}"

    print(header)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'ROC-AUC':<14} {metrics.roc_auc:.4f}")
    print(f"  {'F1':<14} {metrics.f1:.4f}")
    print(f"  {'Precision':<14} {metrics.precision:.4f}")
    print(f"  {'Recall':<14} {metrics.recall:.4f}")
    print(f"  {'F1 (PA)':<14} {metrics.f1_pa:.4f}  ← point-adjust")
    print(f"  {'Prec (PA)':<14} {metrics.precision_pa:.4f}")
    print(f"  {'Recall (PA)':<14} {metrics.recall_pa:.4f}")
    print(f"  {'Threshold':<14} {metrics.best_threshold:.6f}")
    print()
