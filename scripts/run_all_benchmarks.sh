#!/usr/bin/env bash
# run_all_benchmarks.sh — reproduce all benchmark results in one command
#
# Usage:
#   bash scripts/run_all_benchmarks.sh
#
# Optional: download real datasets first with:
#   python datasets/download_datasets.py --datasets ett psm

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================="
echo "  Industrial Time-Series AI — Full Benchmark"
echo "  $(date)"
echo "=============================================="

# 1. Anomaly detection on dummy SWaT
echo ""
echo "[1/3] Anomaly detection (dummy SWaT)"
python benchmarks/run_benchmark.py --task anomaly --epochs 5

# 2. Forecasting on dummy SWaT (all 5 models)
echo ""
echo "[2/3] Forecasting benchmark (dummy SWaT, all models)"
python benchmarks/run_benchmark.py --task forecasting --epochs 5

# 3. Forecasting on real ETTh1 (skipped if not downloaded)
echo ""
echo "[3/3] Forecasting benchmark (ETTh1, all models)"
if [ -f "datasets/raw/ETT/ETTh1.csv" ]; then
    python benchmarks/run_benchmark.py --task forecasting_ett --epochs 10
else
    echo "  [skip] ETTh1 not found. Run:"
    echo "         python datasets/download_datasets.py --datasets ett"
fi

echo ""
echo "All benchmarks complete."
echo "Results saved to benchmarks/results/benchmark_results.csv"
