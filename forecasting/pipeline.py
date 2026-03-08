"""
Forecasting pipeline for Industrial-Time-Series-AI.

Provides a full training, validation, and evaluation loop for sequence-to-sequence
time-series forecasting. Supports:
  - YAML config loading
  - Comprehensive metrics (RMSE, MAE, MAPE, SMAPE)
  - Best-checkpoint saving by validation RMSE
  - Prediction export to CSV
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.forecasting_metrics import (
    compute_forecasting_metrics,
    print_forecasting_metrics,
    print_metrics_comparison_table,
    ForecastingMetrics,
)


class ForecastingPipeline:
    """Training and evaluation loop for industrial time-series forecasting.

    Supports any model that accepts input of shape (B, seq_len, num_features)
    and returns predictions of shape (B, pred_len, num_features) or (B, out_features).

    Args:
        model:          PyTorch model.
        learning_rate:  Initial learning rate for Adam.
        device:         Computation device. Auto-detects GPU if available.
        checkpoint_dir: Directory for saving best checkpoints.
        model_name:     Label used in metric reports and filenames.
        dataset_name:   Label used in metric reports.
        pred_len:       Forecast horizon (used in metric reporting).
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        model_name: str = "model",
        dataset_name: str = "dataset",
        pred_len: Optional[int] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.pred_len = pred_len
        self._best_val_rmse = float("inf")
        self.history: list[dict] = []

    @classmethod
    def from_config(cls, model: nn.Module, config_path: str, **kwargs) -> "ForecastingPipeline":
        """Instantiate from a YAML configuration file.

        Expected YAML structure::

            training:
              learning_rate: 1e-3
              epochs: 20
              checkpoint_dir: checkpoints/
            experiment:
              model_name: PatchTST
              dataset_name: ETTh1
              pred_len: 96
        """
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required for config loading: pip install pyyaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        train_cfg = cfg.get("training", {})
        exp_cfg = cfg.get("experiment", {})
        return cls(
            model=model,
            learning_rate=train_cfg.get("learning_rate", 1e-3),
            checkpoint_dir=train_cfg.get("checkpoint_dir"),
            model_name=exp_cfg.get("model_name", kwargs.get("model_name", "model")),
            dataset_name=exp_cfg.get("dataset_name", kwargs.get("dataset_name", "dataset")),
            pred_len=exp_cfg.get("pred_len", kwargs.get("pred_len")),
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            # Align shape: model may output (B, out) or (B, pred_len, C)
            if predictions.shape != batch_y.shape:
                predictions = predictions.reshape(batch_y.shape)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(
        self, dataloader: DataLoader, split: str = "val"
    ) -> ForecastingMetrics:
        """Evaluate on a dataloader and return full metrics.

        Args:
            dataloader: Validation or test DataLoader.
            split:      Label for reporting ("val" or "test").

        Returns:
            ForecastingMetrics with RMSE, MAE, MAPE, SMAPE.
        """
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x)
                if predictions.shape != batch_y.shape:
                    predictions = predictions.reshape(batch_y.shape)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        return compute_forecasting_metrics(
            targets, preds,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            pred_len=self.pred_len,
        )

    def _save_checkpoint(self, epoch: int) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{self.model_name}_{self.dataset_name}_best.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_rmse": self._best_val_rmse,
            },
            path,
        )

    def load_best_checkpoint(self) -> None:
        """Load the best checkpoint saved during training."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir was not set.")
        path = self.checkpoint_dir / f"{self.model_name}_{self.dataset_name}_best.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

    def save_predictions(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        output_path: str,
    ) -> None:
        """Save predictions and targets to a CSV file.

        Args:
            preds:       Predicted values, shape (N, ...).
            targets:     Ground truth values, shape (N, ...).
            output_path: Path for the CSV file.
        """
        preds_flat = preds.reshape(len(preds), -1)
        targets_flat = targets.reshape(len(targets), -1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            n_out = preds_flat.shape[1]
            header = [f"pred_{i}" for i in range(n_out)] + [f"target_{i}" for i in range(n_out)]
            writer.writerow(header)
            for p, t in zip(preds_flat, targets_flat):
                writer.writerow(list(p) + list(t))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_predictions_path: Optional[str] = None,
    ) -> Optional[ForecastingMetrics]:
        """Train the model.

        Args:
            train_loader:           Training DataLoader.
            val_loader:             Optional validation DataLoader.
            epochs:                 Number of training epochs.
            save_predictions_path:  If set, save final val predictions as CSV.

        Returns:
            Final validation ForecastingMetrics if val_loader is provided, else None.
        """
        print(f"Training {self.model_name} on {self.dataset_name} | device={self.device}")

        final_metrics: Optional[ForecastingMetrics] = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)

            epoch_record: dict = {"epoch": epoch, "train_mse": train_loss}

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, split="val")
                self.scheduler.step(val_metrics.rmse)

                epoch_record.update(val_metrics.to_dict())
                print(
                    f"  Epoch [{epoch}/{epochs}]"
                    f"  Train MSE={train_loss:.4f}"
                    f"  Val RMSE={val_metrics.rmse:.4f}"
                    f"  MAE={val_metrics.mae:.4f}"
                    f"  MAPE={val_metrics.mape:.2f}%"
                )

                if val_metrics.rmse < self._best_val_rmse:
                    self._best_val_rmse = val_metrics.rmse
                    self._save_checkpoint(epoch)

                final_metrics = val_metrics
            else:
                print(f"  Epoch [{epoch}/{epochs}]  Train MSE={train_loss:.4f}")

            self.history.append(epoch_record)

        if final_metrics is not None:
            print("\nFinal validation metrics:")
            print_forecasting_metrics(final_metrics)

            if save_predictions_path and val_loader:
                val_metrics = self.evaluate(val_loader)
                # Re-run to collect arrays
                self.model.eval()
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for bx, by in val_loader:
                        p = self.model(bx.to(self.device))
                        if p.shape != by.shape:
                            p = p.reshape(by.shape)
                        all_preds.append(p.cpu().numpy())
                        all_targets.append(by.numpy())
                self.save_predictions(
                    np.concatenate(all_preds),
                    np.concatenate(all_targets),
                    save_predictions_path,
                )
                print(f"  Predictions saved → {save_predictions_path}")

        return final_metrics
