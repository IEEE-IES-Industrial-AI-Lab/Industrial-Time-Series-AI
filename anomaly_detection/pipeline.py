import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

class ReconstructionAnomalyPipeline:
    """
    Standard training and evaluation loop for Autoencoder-based Anomaly Detection.
    Anomalies are detected based on high reconstruction error.
    """
    def __init__(self, model, learning_rate=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        # Unsupervised training: x is used as both input and target
        for batch_x in dataloader:
            if isinstance(batch_x, (list, tuple)):
                # If dataloader returns (x, y), ignore y during unsupervised training
                batch_x = batch_x[0]
            
            batch_x = batch_x.to(self.device)
            self.optimizer.zero_grad()
            
            reconstruction = self.model(batch_x)
            loss = self.criterion(reconstruction, batch_x)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, threshold=None):
        self.model.eval()
        scores = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Test loaders for anomalies usually yield (X, y) where y is anomaly label
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_x, batch_y = batch[0].to(self.device), batch[1].cpu().numpy()
                    labels.extend(batch_y)
                else:
                    batch_x = batch.to(self.device) if not isinstance(batch, (list, tuple)) else batch[0].to(self.device)
                
                reconstruction = self.model(batch_x)
                
                # Compute reconstruction error per sample in batch
                error = torch.mean((reconstruction - batch_x) ** 2, dim=(1, 2))
                scores.extend(error.cpu().numpy())
                
        scores = np.array(scores)
        metrics = {'anomaly_scores': scores}
        
        if len(labels) > 0:
            labels = np.array(labels)
            try:
                metrics['roc_auc'] = roc_auc_score(labels, scores)
            except ValueError:
                metrics['roc_auc'] = 0.5 # Handle single class in batch
                
            if threshold is not None:
                preds = (scores > threshold).astype(int)
                metrics['f1'] = f1_score(labels, preds)
                metrics['precision'] = precision_score(labels, preds)
                metrics['recall'] = recall_score(labels, preds)
                
        return metrics

    def fit(self, train_loader, epochs=10):
        print(f"Training Autoencoder Anomaly Detector on {self.device}...")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Reconstruction Train MSE: {train_loss:.4f}")
