"""
Anomaly detection pipeline for Industrial-Time-Series-AI.

Provides unsupervised reconstruction-based anomaly detection using autoencoders.
Includes:
  - Standard training loop with MSE reconstruction loss
  - Best-threshold search
  - Point-Adjust (PA) evaluation — de-facto standard for ICS anomaly benchmarks
  - Anomaly score export to CSV
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.anomaly_metrics import (
    compute_anomaly_metrics,
    print_anomaly_metrics,
    AnomalyMetrics,
)


class ReconstructionAnomalyPipeline:
    """Training and evaluation loop for autoencoder-based anomaly detection.

    Anomalies are detected based on reconstruction error: a window with a high
    mean squared reconstruction error is flagged as anomalous.

    The pipeline follows the IEEE IES standard evaluation protocol:
      1. Train on normal (or mixed) data using reconstruction loss.
      2. Compute per-sample reconstruction error on the test set.
      3. Find the optimal threshold via F1 maximisation.
      4. Apply Point-Adjust (PA) — credit entire anomaly segments if any
         point is detected.
      5. Report ROC-AUC, F1, Precision, Recall (raw and PA).

    Args:
        model:          Autoencoder model (must reconstruct input with same shape).
        learning_rate:  Initial learning rate.
        device:         Computation device. Auto-detects GPU if available.
        checkpoint_dir: Directory for saving best checkpoints.
        model_name:     Label used in reports and filenames.
        dataset_name:   Label used in reports and filenames.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        model_name: str = "autoencoder",
        dataset_name: str = "dataset",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss(reduction="none")
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.model_name = model_name
        self.dataset_name = dataset_name
        self._best_train_loss = float("inf")
        self.history: list[dict] = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch on normal (or mixed) data."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_x = batch_x.to(self.device)
            self.optimizer.zero_grad()
            reconstruction = self.model(batch_x)
            loss = self.criterion(reconstruction, batch_x).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def compute_anomaly_scores(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-sample reconstruction errors and collect labels.

        Args:
            dataloader: DataLoader yielding (x,) or (x, y) batches.

        Returns:
            (scores, labels) — both 1-D arrays of shape (N,).
            labels is all-zeros if the dataloader has no labels.
        """
        self.model.eval()
        scores_list, labels_list = [], []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_x = batch[0].to(self.device)
                    labels_list.extend(batch[1].cpu().numpy().tolist())
                else:
                    batch_x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)

                recon = self.model(batch_x)
                # Per-sample mean reconstruction error across (time, features) dims
                error = self.criterion(recon, batch_x).mean(dim=(1, 2))
                scores_list.extend(error.cpu().numpy().tolist())

        scores = np.array(scores_list, dtype=float)
        labels = np.array(labels_list, dtype=int) if labels_list else np.zeros(len(scores), dtype=int)
        return scores, labels

    def evaluate(
        self,
        dataloader: DataLoader,
        threshold: Optional[float] = None,
    ) -> AnomalyMetrics:
        """Evaluate anomaly detection performance on a labelled test set.

        Args:
            dataloader: Test DataLoader that returns (x, label) pairs.
            threshold:  Fixed decision threshold. If None, best F1 threshold is found.

        Returns:
            AnomalyMetrics with ROC-AUC, F1, Precision, Recall (raw + PA).
        """
        scores, labels = self.compute_anomaly_scores(dataloader)
        return compute_anomaly_metrics(
            y_true=labels,
            scores=scores,
            threshold=threshold,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )

    def save_anomaly_scores(
        self, scores: np.ndarray, labels: np.ndarray, output_path: str
    ) -> None:
        """Save anomaly scores and labels to CSV for offline analysis.

        Args:
            scores:      Per-sample anomaly scores, shape (N,).
            labels:      Ground truth binary labels, shape (N,).
            output_path: Destination CSV path.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "anomaly_score", "true_label"])
            for i, (s, l) in enumerate(zip(scores, labels)):
                writer.writerow([i, float(s), int(l)])

    def _save_checkpoint(self, epoch: int) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{self.model_name}_{self.dataset_name}_best.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_train_loss": self._best_train_loss,
            },
            path,
        )

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_scores_path: Optional[str] = None,
    ) -> Optional[AnomalyMetrics]:
        """Train the autoencoder and optionally evaluate on a test set.

        Args:
            train_loader:     Training DataLoader (normal or mixed data).
            test_loader:      Optional labelled test DataLoader for evaluation.
            epochs:           Number of training epochs.
            save_scores_path: If set, save anomaly scores to CSV after evaluation.

        Returns:
            AnomalyMetrics from the test set (if test_loader provided), else None.
        """
        print(f"Training {self.model_name} on {self.dataset_name} | device={self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.scheduler.step(train_loss)
            print(f"  Epoch [{epoch}/{epochs}]  Recon MSE={train_loss:.4f}")

            if train_loss < self._best_train_loss:
                self._best_train_loss = train_loss
                self._save_checkpoint(epoch)

            self.history.append({"epoch": epoch, "train_recon_mse": train_loss})

        if test_loader is None:
            return None

        print("\nEvaluating on test set…")
        scores, labels = self.compute_anomaly_scores(test_loader)
        metrics = compute_anomaly_metrics(
            y_true=labels,
            scores=scores,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )
        print_anomaly_metrics(metrics)

        if save_scores_path:
            self.save_anomaly_scores(scores, labels, save_scores_path)
            print(f"  Anomaly scores saved → {save_scores_path}")

        return metrics
