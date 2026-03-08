import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

class ForecastingPipeline:
    """
    Standard training and evaluation loop for Industrial Time-Series Forecasting.
    """
    def __init__(self, model, learning_rate=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward
            predictions = self.model(batch_x)
            
            # Compute Loss
            loss = self.criterion(predictions, batch_y)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
        metrics = {
            'mse': total_loss / len(dataloader),
            'preds': np.concatenate(all_preds, axis=0),
            'targets': np.concatenate(all_targets, axis=0)
        }
        return metrics

    def fit(self, train_loader, val_loader=None, epochs=10):
        print(f"Training on device: {self.device}")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_msg = ""
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_msg = f" | Val MSE: {val_metrics['mse']:.4f}"
            print(f"Epoch [{epoch+1}/{epochs}] - Train MSE: {train_loss:.4f}{val_msg}")
