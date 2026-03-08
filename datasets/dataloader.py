import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class IndustrialTimeSeriesDataset(Dataset):
    """
    Base class for Industrial Time Series Datasets (e.g., SWaT, WADI).
    In real scenarios, this would load from CSV or Parquet files.
    Here we provide a dummy generator for testing and structural validation.
    """
    def __init__(self, data, labels=None, window_size=100, step_size=1, is_train=True):
        """
        Args:
            data (np.ndarray): Shape (num_samples, num_features).
            labels (np.ndarray, optional): Shape (num_samples,). Missing if forecasting.
            window_size (int): Length of the sliding window.
            step_size (int): Stride for the sliding window.
            is_train (bool): Flag indicating if this is the training set (used for scaling logic).
        """
        super().__init__()
        self.data = np.array(data, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32) if labels is not None else None
        self.window_size = window_size
        self.step_size = step_size
        self.is_train = is_train

        # Built-in sanity scaling (typically done globally before windowing, but shown here for integrity)
        self.scaler = StandardScaler()
        if self.is_train:
            self.data = self.scaler.fit_transform(self.data)
        else:
            # Note: In practice, we'd load the fitted scaler from train
            self.data = self.scaler.fit_transform(self.data)
        
        self.windows = self._create_windows()

    def _create_windows(self):
        """Generates window indices based on data length, window_size, and step_size."""
        num_windows = (len(self.data) - self.window_size) // self.step_size + 1
        return [i * self.step_size for i in range(num_windows)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_idx = self.windows[idx]
        end_idx = start_idx + self.window_size
        
        x = self.data[start_idx:end_idx]
        x_tensor = torch.tensor(x, dtype=torch.float32)

        if self.labels is not None:
            # For anomaly detection, the label of the window is often the label of the last point
            # or if any point in the window is anomalous. Let's use max for anomaly.
            y = np.max(self.labels[start_idx:end_idx])
            y_tensor = torch.tensor(int(y), dtype=torch.long)
            return x_tensor, y_tensor
        
        return x_tensor

def get_dummy_swat_dataloader(batch_size=32, window_size=100, num_samples=10000, num_features=51):
    """Generates a dummy DataLoader representing the SWaT dataset."""
    print("Generating Mock SWaT Dataset...")
    dummy_data = np.random.randn(num_samples, num_features)
    # Inject synthetic anomalies
    dummy_labels = np.zeros(num_samples)
    anomaly_indices = np.random.choice(num_samples, size=int(0.05 * num_samples), replace=False)
    dummy_labels[anomaly_indices] = 1
    dummy_data[anomaly_indices] += 3.0 * np.random.randn(num_features) # Shift distribution

    dataset = IndustrialTimeSeriesDataset(dummy_data, dummy_labels, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def get_dummy_wadi_dataloader(batch_size=32, window_size=100, num_samples=10000, num_features=123):
    """Generates a dummy DataLoader representing the WADI dataset."""
    print("Generating Mock WADI Dataset...")
    dummy_data = np.random.randn(num_samples, num_features)
    dummy_labels = np.zeros(num_samples)
    dataset = IndustrialTimeSeriesDataset(dummy_data, dummy_labels, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
