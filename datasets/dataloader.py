"""
Dataset loaders for Industrial-Time-Series-AI.

Provides:
  - IndustrialTimeSeriesDataset  — generic sliding-window dataset
  - ETTDataset                   — Electricity Transformer Temperature (ETTh1/h2/m1/m2)
  - PSMDataset                   — Pooled Server Metrics anomaly detection dataset
  - get_dataset()                — unified factory function
  - get_dummy_swat_dataloader()  — synthetic SWaT-like anomaly data (no download needed)
  - get_dummy_wadi_dataloader()  — synthetic WADI-like data (no download needed)

Real datasets can be downloaded with:
    python datasets/download_datasets.py --datasets ett psm
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Literal

RAW_DIR = Path(__file__).parent / "raw"


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

class IndustrialTimeSeriesDataset(Dataset):
    """Sliding-window dataset for industrial time-series.

    Wraps a raw numpy array, applies standard scaling, and creates
    overlapping windows for use with all models in this repo.

    Args:
        data:        Raw array of shape (T, num_features).
        labels:      Optional anomaly labels of shape (T,).
        window_size: Temporal window length.
        step_size:   Stride between consecutive windows.
        scaler:      Pre-fitted StandardScaler (from training split).
                     If None and is_train=True, a new scaler is fitted.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        window_size: int = 100,
        step_size: int = 1,
        is_train: bool = True,
        scaler: Optional[StandardScaler] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.step_size = step_size

        data = np.asarray(data, dtype=np.float32)

        if scaler is None:
            self.scaler = StandardScaler()
            if is_train:
                data = self.scaler.fit_transform(data)
            else:
                # Fallback: fit on this split (caller should pass train scaler)
                data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            data = self.scaler.transform(data)

        self.data = data
        self.labels = np.asarray(labels, dtype=np.float32) if labels is not None else None
        self._window_starts = list(
            range(0, len(data) - window_size + 1, step_size)
        )

    def __len__(self) -> int:
        return len(self._window_starts)

    def __getitem__(self, idx: int):
        start = self._window_starts[idx]
        x = torch.tensor(self.data[start : start + self.window_size], dtype=torch.float32)

        if self.labels is not None:
            y = int(np.max(self.labels[start : start + self.window_size]))
            return x, torch.tensor(y, dtype=torch.long)

        return x


# ---------------------------------------------------------------------------
# ETT Dataset (real data)
# ---------------------------------------------------------------------------

class ETTDataset(Dataset):
    """Electricity Transformer Temperature dataset for forecasting.

    Supports ETTh1, ETTh2 (hourly) and ETTm1, ETTm2 (15-minute) variants.
    Download first: python datasets/download_datasets.py --datasets ett

    Standard train/val/test splits follow the Time-Series-Library convention:
      ETTh: train=8640, val=2880, test=2880
      ETTm: train=34560, val=11520, test=11520

    Args:
        name:      One of "ETTh1", "ETTh2", "ETTm1", "ETTm2".
        split:     "train", "val", or "test".
        seq_len:   Lookback window length (default 96).
        pred_len:  Forecast horizon (default 96).
        target:    Column name to forecast. If "all", all 7 features are used.
        data_path: Override for the CSV file path.
    """

    _SPLIT_SIZES = {
        "ETTh1": (8640, 2880, 2880),
        "ETTh2": (8640, 2880, 2880),
        "ETTm1": (34560, 11520, 11520),
        "ETTm2": (34560, 11520, 11520),
    }

    def __init__(
        self,
        name: str = "ETTh1",
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = 96,
        pred_len: int = 96,
        target: str = "OT",
        data_path: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        csv_path = Path(data_path) if data_path else RAW_DIR / "ETT" / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"ETT file not found at {csv_path}.\n"
                "Run: python datasets/download_datasets.py --datasets ett"
            )

        df = pd.read_csv(csv_path)
        df = df.drop(columns=["date"], errors="ignore")

        total_len = len(df)
        train_size, val_size, test_size = self._SPLIT_SIZES.get(name, (int(total_len * 0.7), int(total_len * 0.1), int(total_len * 0.2)))

        if split == "train":
            border = (0, train_size)
        elif split == "val":
            border = (train_size - seq_len, train_size + val_size)
        else:
            border = (train_size + val_size - seq_len, train_size + val_size + test_size)

        raw = df.values.astype(np.float32)

        # Fit scaler on training portion only
        self.scaler = StandardScaler()
        self.scaler.fit(raw[:train_size])
        data = self.scaler.transform(raw)

        self.data_x = data[border[0] : border[1]]
        self.data_y = data[border[0] : border[1]]

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data_x[idx : idx + self.seq_len]
        y = self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# PSM Dataset (real data, anomaly detection)
# ---------------------------------------------------------------------------

class PSMDataset(Dataset):
    """Pooled Server Metrics (PSM) anomaly detection dataset.

    25-dimensional server resource metrics from Microsoft.
    Download first: python datasets/download_datasets.py --datasets psm

    Args:
        split:     "train" or "test".
        seq_len:   Sliding window length (default 100).
        step_size: Window stride (default 1).
        data_path: Override for the data directory path.
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        seq_len: int = 100,
        step_size: int = 1,
        data_path: Optional[str] = None,
    ):
        psm_dir = Path(data_path) if data_path else RAW_DIR / "PSM"
        data_file = psm_dir / f"{split}.csv"
        label_file = psm_dir / "test_label.csv" if split == "test" else None

        if not data_file.exists():
            raise FileNotFoundError(
                f"PSM file not found at {data_file}.\n"
                "Run: python datasets/download_datasets.py --datasets psm"
            )

        df = pd.read_csv(data_file)
        df = df.drop(columns=["timestamp_(min)"], errors="ignore")
        data = df.values.astype(np.float32)

        scaler = StandardScaler()
        if split == "train":
            data = scaler.fit_transform(data)
            labels = None
        else:
            train_file = psm_dir / "train.csv"
            if train_file.exists():
                train_df = pd.read_csv(train_file).drop(columns=["timestamp_(min)"], errors="ignore")
                scaler.fit(train_df.values.astype(np.float32))
            data = scaler.transform(data)
            if label_file and label_file.exists():
                label_df = pd.read_csv(label_file)
                labels = label_df.values[:, -1].astype(np.float32)
            else:
                labels = np.zeros(len(data), dtype=np.float32)

        self.dataset = IndustrialTimeSeriesDataset(
            data=data,
            labels=labels if split == "test" else None,
            window_size=seq_len,
            step_size=step_size,
            is_train=(split == "train"),
            scaler=scaler,
        )
        # Bypass double-scaling: dataset was already scaled above
        self.dataset.data = data
        self.dataset.scaler = scaler

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


# ---------------------------------------------------------------------------
# Dummy loaders (no download required)
# ---------------------------------------------------------------------------

def get_dummy_swat_dataloader(
    batch_size: int = 32,
    window_size: int = 100,
    num_samples: int = 10000,
    num_features: int = 51,
) -> DataLoader:
    """Synthetic SWaT-like DataLoader for quick testing.

    Generates 51-feature multivariate data with ~5% anomaly segments
    injected as distribution shifts (+3σ). Labels are provided for
    anomaly detection evaluation.
    """
    print("Generating Mock SWaT Dataset...")
    rng = np.random.default_rng(42)
    data = rng.standard_normal((num_samples, num_features)).astype(np.float32)

    labels = np.zeros(num_samples, dtype=np.float32)
    # Inject contiguous anomaly segments (more realistic than random points)
    seg_len = 50
    num_segs = max(1, int(0.05 * num_samples / seg_len))
    starts = rng.integers(0, num_samples - seg_len, size=num_segs)
    for s in starts:
        data[s : s + seg_len] += 3.0 * rng.standard_normal((seg_len, num_features))
        labels[s : s + seg_len] = 1.0

    dataset = IndustrialTimeSeriesDataset(data, labels, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def get_dummy_wadi_dataloader(
    batch_size: int = 32,
    window_size: int = 100,
    num_samples: int = 10000,
    num_features: int = 123,
) -> DataLoader:
    """Synthetic WADI-like DataLoader (no anomaly labels)."""
    print("Generating Mock WADI Dataset...")
    rng = np.random.default_rng(0)
    data = rng.standard_normal((num_samples, num_features)).astype(np.float32)
    labels = np.zeros(num_samples, dtype=np.float32)
    dataset = IndustrialTimeSeriesDataset(data, labels, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_dataset(
    name: str,
    split: str = "train",
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    step_size: int = 1,
    data_path: Optional[str] = None,
) -> DataLoader:
    """Unified factory for all datasets in this repo.

    Args:
        name:      One of "ETTh1", "ETTh2", "ETTm1", "ETTm2", "PSM",
                   "swat_dummy", "wadi_dummy".
        split:     "train", "val", or "test".
        seq_len:   Input sequence / window length.
        pred_len:  Forecast horizon (ETT only).
        batch_size: DataLoader batch size.
        step_size: Window stride (PSM / industrial datasets).
        data_path: Optional override for the data file/directory.

    Returns:
        A configured DataLoader.
    """
    name_lower = name.lower()

    if name_lower in ("etth1", "etth2", "ettm1", "ettm2"):
        canonical = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2"}[name_lower]
        dataset = ETTDataset(
            name=canonical,
            split=split,
            seq_len=seq_len,
            pred_len=pred_len,
            data_path=data_path,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), drop_last=True)

    if name_lower == "psm":
        dataset = PSMDataset(split=split, seq_len=seq_len, step_size=step_size, data_path=data_path)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), drop_last=True)

    if name_lower == "swat_dummy":
        return get_dummy_swat_dataloader(batch_size=batch_size, window_size=seq_len)

    if name_lower == "wadi_dummy":
        return get_dummy_wadi_dataloader(batch_size=batch_size, window_size=seq_len)

    raise ValueError(
        f"Unknown dataset '{name}'. Choose from: ETTh1, ETTh2, ETTm1, ETTm2, PSM, swat_dummy, wadi_dummy."
    )
