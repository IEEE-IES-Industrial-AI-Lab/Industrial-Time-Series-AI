import argparse
import sys
import os

# Ensure the root directory is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataloader import get_dummy_swat_dataloader
from models.lstm_forecasting import LSTMAutoencoder
from anomaly_detection.pipeline import ReconstructionAnomalyPipeline

def run_swat_anomaly_benchmark():
    print("="*50)
    print(" Industrial Time-Series AI Benchmark: SWaT Anomaly")
    print("="*50)
    
    # 1. Load Data
    print("\n[1/3] Initializing Data Loaders...")
    train_loader = get_dummy_swat_dataloader(batch_size=64, num_samples=5000)
    test_loader = get_dummy_swat_dataloader(batch_size=64, num_samples=2000)
    
    # 2. Initialize Model
    print("\n[2/3] Initializing LSTM Autoencoder...")
    # SWaT typically has 51 features
    model = LSTMAutoencoder(num_features=51, hidden_dim=32, num_layers=1)
    
    # 3. Train & Evaluate Pipeline
    print("\n[3/3] Starting Training Pipeline...")
    pipeline = ReconstructionAnomalyPipeline(model, learning_rate=1e-3)
    
    # Fit unsupervised on train data
    pipeline.fit(train_loader, epochs=3)
    
    # Evaluate on test data
    print("\nEvaluating on Test Set...")
    metrics = pipeline.evaluate(test_loader)
    
    print("\nBenchmark Results:")
    print(f"ROC-AUC Score: {metrics.get('roc_auc', 'N/A'):.4f}")
    if 'f1' in metrics:
        print(f"F1-Score: {metrics['f1']:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks for Industrial Time-Series AI")
    parser.add_argument("--task", type=str, default="swat_anomaly", help="Task to run (e.g., swat_anomaly)")
    args = parser.parse_args()
    
    if args.task == "swat_anomaly":
        run_swat_anomaly_benchmark()
    else:
        print(f"Unknown task: {args.task}")
