# Industrial Time-Series AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, benchmarking AI framework designed specifically for multivariate industrial sensor streams. This repository implements state-of-the-art (SOTA) deep learning models tailored for industrial time-series analysis, facilitating both forecasting and anomaly detection on challenging real-world datasets.

## Why Industrial Time-Series AI?

Factories and modern industrial systems generate massive streams of multivariate sensor data continuously. Most generic machine learning libraries overlook the unique, complex characteristics of industrial time-series data, such as:
- Extreme class imbalance (rare failures)
- Multi-scale temporal dependencies
- High-dimensional spatial-temporal correlations

This repository bridges that gap by providing a robust, highly modular framework for researching and deploying AI on industrial data.

## Key Features

*   **SOTA Architectures**: Ready-to-use implementations of LSTM, TCN, Time-Series Transformers, and PatchTST.
*   **Dual Pipelines**: Unified workflows for both *Forecasting* (predicting future states) and *Anomaly Detection* (identifying abnormal behaviors).
*   **Industrial Benchmark Compatibility**: Built-in support (via loaders) for prominent industrial datasets like SWaT (Secure Water Treatment) and WADI (Water Distribution).
*   **Advanced Feature Engineering**: Tools for robust sliding window generation and temporal feature extraction.
*   **Visualization Suite**: Utilities to plot complex multi-channel sensor data alongside model predictions and anomaly scores.

## Repository Structure

```
industrial-time-series-ai/
├── anomaly_detection/   # End-to-end anomaly detection training & evaluation
├── benchmarks/          # Scripts to reproduce standard benchmark results
├── datasets/            # Data loaders for SWaT, WADI, and custom datasets
├── feature_engineering/ # Time-series feature extraction and windowing strategies
├── forecasting/         # End-to-end forecasting training & evaluation
├── models/              # Core PyTorch implementations of SOTA models
├── tutorials/           # Jupyter notebooks for quickstart and experimentation
└── visualization/       # Plotting and inspection utilities
```

## Supported Models

| Model | Type | Architecture Highlights | Best For |
| :--- | :--- | :--- | :--- |
| **LSTM** | Recurrent | Deep Recurrent Network, Sequential Memory | Baseline sequence learning |
| **TCN** | Convolutional | Causal Convolutions, Dilations, Residual Blocks | Fast, parallelizable sequence modeling |
| **Transformer** | Attention | Self-Attention Mechanism, Positional Encoding | Long-range dependencies |
| **PatchTST** | Attention | Patching, Channel-Independence, Advanced Attention | SOTA long-term forecasting |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IEEE-IES-Industrial-AI-Lab/Industrial-Time-Series-AI.git
   cd Industrial-Time-Series-AI
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quickstart Tutorial

Check out our tutorials to get started right away:
- [Tutorial 1: Forecasting with PatchTST](tutorials/01_forecasting_with_patchtst.ipynb)

## Contributing

We welcome contributions from researchers and engineers! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
