import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

def plot_multivariate_timeseries(data, feature_names=None, title="Multivariate Time Series", max_subplots=5):
    """
    Plots a multivariate time-series.
    Args:
        data (np.ndarray): Shape (seq_len, num_features)
    """
    seq_len, num_features = data.shape
    num_plots = min(num_features, max_subplots)
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 2 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
        
    for i in range(num_plots):
        axes[i].plot(data[:, i], lw=1.5, alpha=0.8)
        name = feature_names[i] if feature_names else f"Sensor {i+1}"
        axes[i].set_ylabel(name, fontsize=10)
        
    axes[-1].set_xlabel("Time (steps)", fontsize=12)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig

def plot_anomaly_scores(true_data, anomaly_scores, labels, threshold=None, feature_idx=0):
    """
    Plots one representative feature alongside the computed anomaly score and ground truth.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Plot true data (Sensor reading)
    ax1.plot(true_data[:, feature_idx], color='blue', alpha=0.7, label='Sensor Reading')
    
    # Highlight anomalies in ground truth
    anom_indices = np.where(labels == 1)[0]
    ax1.scatter(anom_indices, true_data[anom_indices, feature_idx], color='red', s=10, label='True Anomalies')
    ax1.set_ylabel("Sensor Value")
    ax1.legend(loc='upper right')
    ax1.set_title("Industrial Signal with Anomalies")
    
    # Plot Anomaly Score
    ax2.plot(anomaly_scores, color='orange', label='Reconstruction Error')
    if threshold is not None:
        ax2.axhline(threshold, color='red', linestyle='--', label='Detection Threshold')
        
    ax2.set_xlabel("Time (steps)")
    ax2.set_ylabel("Anomaly Score")
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig
