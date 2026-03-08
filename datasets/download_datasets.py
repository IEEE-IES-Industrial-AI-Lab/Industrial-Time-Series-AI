"""
Dataset downloader for Industrial-Time-Series-AI.

Downloads publicly available industrial time-series datasets into
the datasets/raw/ directory. Supported datasets:

  ETT (Electricity Transformer Temperature)
    - ETTh1, ETTh2  — hourly, 7 features, 17420 steps
    - ETTm1, ETTm2  — 15-min, 7 features, 69680 steps
    Source: https://github.com/zhouhaoyi/ETDataset

  PSM (Pooled Server Metrics) — anomaly detection
    - 25 features, 132481 train + 87841 test steps
    Source: https://github.com/NetManAIOps/OmniAnomaly (MSR)

  SMAP / MSL — NASA anomaly datasets
    - Multi-channel telemetry with labelled anomalies
    Source: https://github.com/khundman/telemanom

Usage:
    python datasets/download_datasets.py --all
    python datasets/download_datasets.py --datasets ett psm
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

ETT_BASE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
)
ETT_FILES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]

PSM_BASE_URL = (
    "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/PSM/"
)
PSM_FILES = ["train.csv", "test.csv", "test_label.csv"]

SMAP_MSL_URL = (
    "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
)


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        print(f"\r    {pct:5.1f}%  ({downloaded // 1024} KB / {total_size // 1024} KB)", end="", flush=True)


def _download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()
        print(f"  Saved → {dest}")
    except Exception as exc:
        print(f"\n  [ERROR] Failed to download {url}: {exc}")
        if dest.exists():
            dest.unlink()


def download_ett() -> None:
    """Download ETTh1, ETTh2, ETTm1, ETTm2 CSV files."""
    print("\n=== ETT (Electricity Transformer Temperature) ===")
    ett_dir = RAW_DIR / "ETT"
    for fname in ETT_FILES:
        _download_file(ETT_BASE_URL + fname, ett_dir / fname)
    print("  ETT download complete.")


def download_psm() -> None:
    """Download PSM (Pooled Server Metrics) anomaly detection dataset."""
    print("\n=== PSM (Pooled Server Metrics) ===")
    psm_dir = RAW_DIR / "PSM"
    for fname in PSM_FILES:
        _download_file(PSM_BASE_URL + fname, psm_dir / fname)
    print("  PSM download complete.")


def download_smap_msl() -> None:
    """Download SMAP / MSL NASA telemetry datasets."""
    print("\n=== SMAP / MSL (NASA Telemetry) ===")
    smap_dir = RAW_DIR / "SMAP_MSL"
    zip_path = smap_dir / "data.zip"
    smap_dir.mkdir(parents=True, exist_ok=True)

    _download_file(SMAP_MSL_URL, zip_path)

    if zip_path.exists():
        print("  Extracting archive…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(smap_dir)
        zip_path.unlink()
        print("  SMAP/MSL extraction complete.")


def check_datasets() -> None:
    """Print a summary of what is already downloaded."""
    print("\n=== Dataset Status ===")
    datasets = {
        "ETT": [RAW_DIR / "ETT" / f for f in ETT_FILES],
        "PSM": [RAW_DIR / "PSM" / f for f in PSM_FILES],
        "SMAP_MSL": [RAW_DIR / "SMAP_MSL" / "data"],
    }
    for name, paths in datasets.items():
        found = sum(1 for p in paths if p.exists())
        total = len(paths)
        status = "OK" if found == total else f"{found}/{total} files"
        print(f"  {name:<12} {status}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download industrial time-series datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ett", "psm", "smap_msl"],
        help="Specific datasets to download.",
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets.")
    parser.add_argument("--check", action="store_true", help="Show download status.")
    args = parser.parse_args()

    if args.check:
        check_datasets()
        return

    if not args.all and not args.datasets:
        parser.print_help()
        return

    targets = set(args.datasets or [])
    if args.all:
        targets = {"ett", "psm", "smap_msl"}

    if "ett" in targets:
        download_ett()
    if "psm" in targets:
        download_psm()
    if "smap_msl" in targets:
        download_smap_msl()

    check_datasets()


if __name__ == "__main__":
    main()
