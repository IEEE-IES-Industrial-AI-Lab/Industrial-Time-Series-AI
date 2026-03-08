"""
Industrial Time-Series AI — Benchmark Runner
=============================================

Runs multi-model, multi-dataset benchmarks and saves results to
benchmarks/results/benchmark_results.csv.

Usage:
    # Anomaly detection benchmark (no download required):
    python benchmarks/run_benchmark.py --task anomaly

    # Forecasting benchmark on dummy SWaT data (no download required):
    python benchmarks/run_benchmark.py --task forecasting

    # Forecasting benchmark on real ETTh1 (download first):
    python benchmarks/run_benchmark.py --task forecasting_ett

    # Run all benchmarks:
    python benchmarks/run_benchmark.py --task all
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from datasets.dataloader import get_dataset
from models.lstm_forecasting import LSTMForecaster, LSTMAutoencoder
from models.tcn_model import TCNForecaster
from models.transformer_ts import TimeSeriesTransformer
from models.patchtst import PatchTST
from models.dlinear import DLinear
from forecasting.pipeline import ForecastingPipeline
from anomaly_detection.pipeline import ReconstructionAnomalyPipeline
from visualization.plot_utils import print_benchmark_table

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "benchmark_results.csv"


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_forecasting_models(num_features: int, seq_len: int, pred_len: int) -> dict:
    """Instantiate all five forecasting models with comparable capacity."""
    return {
        "LSTM": LSTMForecaster(
            num_features=num_features,
            hidden_dim=64,
            num_layers=2,
            out_features=pred_len * num_features,
        ),
        "TCN": TCNForecaster(
            num_features=num_features,
            num_channels=[32, 64, 64],
            kernel_size=3,
            out_features=pred_len * num_features,
        ),
        "Transformer": TimeSeriesTransformer(
            num_features=num_features,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            out_features=pred_len * num_features,
        ),
        "PatchTST": PatchTST(
            num_features=num_features,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=16,
            stride=8,
            d_model=64,
            nhead=4,
            num_layers=2,
        ),
        "DLinear": DLinear(
            seq_len=seq_len,
            pred_len=pred_len,
            num_features=num_features,
            kernel_size=25,
            individual=True,
        ),
    }


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def append_results(rows: list[dict]) -> None:
    """Append result rows to the benchmark CSV file."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Forecasting benchmark (dummy SWaT)
# ---------------------------------------------------------------------------

def run_forecasting_benchmark(
    dataset_name: str = "swat_dummy",
    num_features: int = 51,
    seq_len: int = 96,
    pred_len: int = 24,
    epochs: int = 5,
    batch_size: int = 32,
    num_samples: int = 4000,
) -> list[dict]:
    """Run all forecasting models on a dataset and return results."""
    print(f"\n{'='*60}")
    print(f"  FORECASTING BENCHMARK  |  dataset={dataset_name}  pred_len={pred_len}")
    print(f"{'='*60}\n")

    # Build dummy forecasting loaders that return (x, y) pairs
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((num_samples + seq_len + pred_len, num_features)).astype("float32")
    xs, ys = [], []
    for i in range(num_samples):
        xs.append(raw[i : i + seq_len])
        ys.append(raw[i + seq_len : i + seq_len + pred_len])
    X = torch.tensor(np.array(xs))
    Y = torch.tensor(np.array(ys))  # (N, pred_len, C)
    split = int(0.8 * num_samples)
    train_ds = TensorDataset(X[:split], Y[:split])
    val_ds = TensorDataset(X[split:], Y[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    models = build_forecasting_models(num_features, seq_len, pred_len)
    results = []

    for model_name, model in models.items():
        print(f"--- {model_name} ---")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()

        pipeline = ForecastingPipeline(
            model=model,
            learning_rate=1e-3,
            model_name=model_name,
            dataset_name=dataset_name,
            pred_len=pred_len,
        )
        metrics = pipeline.fit(train_loader, val_loader, epochs=epochs)
        elapsed = time.time() - t0

        row = {
            "task": "forecasting",
            "model": model_name,
            "dataset": dataset_name,
            "pred_len": pred_len,
            "params": n_params,
            "train_time_s": round(elapsed, 1),
            "RMSE": round(metrics.rmse, 4) if metrics else "N/A",
            "MAE": round(metrics.mae, 4) if metrics else "N/A",
            "MAPE": round(metrics.mape, 2) if metrics else "N/A",
            "SMAPE": round(metrics.smape, 2) if metrics else "N/A",
        }
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# ETT Forecasting benchmark (requires download)
# ---------------------------------------------------------------------------

def run_forecasting_ett_benchmark(
    dataset_name: str = "ETTh1",
    seq_len: int = 96,
    pred_len: int = 96,
    epochs: int = 10,
    batch_size: int = 32,
) -> list[dict]:
    """Run forecasting benchmark on real ETTh1 data."""
    print(f"\n{'='*60}")
    print(f"  ETT FORECASTING BENCHMARK  |  dataset={dataset_name}  pred_len={pred_len}")
    print(f"{'='*60}\n")

    try:
        train_loader = get_dataset(dataset_name, split="train", seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
        val_loader = get_dataset(dataset_name, split="val", seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Tip: run  python datasets/download_datasets.py --datasets ett  first.")
        return []

    # ETT has 7 features
    num_features = 7
    models = build_forecasting_models(num_features, seq_len, pred_len)
    results = []

    for model_name, model in models.items():
        print(f"--- {model_name} ---")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()

        pipeline = ForecastingPipeline(
            model=model,
            learning_rate=1e-3,
            model_name=model_name,
            dataset_name=dataset_name,
            pred_len=pred_len,
            checkpoint_dir=str(RESULTS_DIR / "checkpoints"),
        )
        metrics = pipeline.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            save_predictions_path=str(RESULTS_DIR / f"{model_name}_{dataset_name}_preds.csv"),
        )
        elapsed = time.time() - t0

        row = {
            "task": "forecasting",
            "model": model_name,
            "dataset": dataset_name,
            "pred_len": pred_len,
            "params": n_params,
            "train_time_s": round(elapsed, 1),
            "RMSE": round(metrics.rmse, 4) if metrics else "N/A",
            "MAE": round(metrics.mae, 4) if metrics else "N/A",
            "MAPE": round(metrics.mape, 2) if metrics else "N/A",
            "SMAPE": round(metrics.smape, 2) if metrics else "N/A",
        }
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Anomaly detection benchmark (dummy SWaT)
# ---------------------------------------------------------------------------

def run_swat_anomaly_benchmark(
    epochs: int = 5,
    batch_size: int = 32,
    num_samples: int = 8000,
    num_features: int = 51,
    window_size: int = 100,
) -> list[dict]:
    """Run autoencoder anomaly detection on synthetic SWaT data."""
    print(f"\n{'='*60}")
    print(f"  ANOMALY DETECTION BENCHMARK  |  dataset=SWaT (dummy)")
    print(f"{'='*60}\n")

    from datasets.dataloader import get_dummy_swat_dataloader

    train_loader = get_dummy_swat_dataloader(
        batch_size=batch_size,
        window_size=window_size,
        num_samples=num_samples,
        num_features=num_features,
    )
    test_loader = get_dummy_swat_dataloader(
        batch_size=batch_size,
        window_size=window_size,
        num_samples=2000,
        num_features=num_features,
    )

    model = LSTMAutoencoder(
        num_features=num_features,
        hidden_dim=64,
        num_layers=2,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.time()

    pipeline = ReconstructionAnomalyPipeline(
        model=model,
        learning_rate=1e-3,
        model_name="LSTMAutoencoder",
        dataset_name="SWaT_dummy",
    )
    metrics = pipeline.fit(
        train_loader,
        test_loader=test_loader,
        epochs=epochs,
        save_scores_path=str(RESULTS_DIR / "swat_anomaly_scores.csv"),
    )
    elapsed = time.time() - t0

    row = {
        "task": "anomaly_detection",
        "model": "LSTMAutoencoder",
        "dataset": "SWaT_dummy",
        "pred_len": "-",
        "params": n_params,
        "train_time_s": round(elapsed, 1),
        "ROC-AUC": round(metrics.roc_auc, 4) if metrics else "N/A",
        "F1": round(metrics.f1, 4) if metrics else "N/A",
        "F1-PA": round(metrics.f1_pa, 4) if metrics else "N/A",
        "Precision": round(metrics.precision, 4) if metrics else "N/A",
        "Recall": round(metrics.recall, 4) if metrics else "N/A",
    }
    return [row]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Industrial Time-Series AI benchmarks."
    )
    parser.add_argument(
        "--task",
        choices=["anomaly", "forecasting", "forecasting_ett", "all"],
        default="anomaly",
        help=(
            "anomaly       — LSTMAutoencoder on dummy SWaT\n"
            "forecasting   — 5 models on dummy SWaT (no download needed)\n"
            "forecasting_ett — 5 models on real ETTh1 (download first)\n"
            "all           — run all of the above"
        ),
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    args = parser.parse_args()

    all_results: list[dict] = []

    if args.task in ("anomaly", "all"):
        rows = run_swat_anomaly_benchmark(epochs=args.epochs or 5)
        all_results.extend(rows)

    if args.task in ("forecasting", "all"):
        rows = run_forecasting_benchmark(epochs=args.epochs or 5)
        all_results.extend(rows)

    if args.task in ("forecasting_ett", "all"):
        rows = run_forecasting_ett_benchmark(epochs=args.epochs or 10)
        all_results.extend(rows)

    if all_results:
        append_results(all_results)
        print("\n" + "=" * 60)
        print("  BENCHMARK SUMMARY")
        print("=" * 60)
        print_benchmark_table(all_results)
        print(f"  Full results saved → {RESULTS_CSV}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
