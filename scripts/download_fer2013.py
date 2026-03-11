#!/usr/bin/env python3
"""
Download FER2013 dataset for training.
Uses Kaggle API - run: pip install opendatasets
Configure: https://github.com/Kaggle/kaggle-api#api-credentials
"""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data", help="Output directory")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    fer_dir = data_dir / "fer2013"
    fer_dir.mkdir(parents=True, exist_ok=True)

    if (fer_dir / "fer2013.csv").exists():
        print("fer2013.csv already exists. Skipping download.")
        return 0

    try:
        import opendatasets as od
    except ImportError:
        print("Install opendatasets: pip install opendatasets")
        print("Then configure Kaggle API: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)

    print("Downloading FER2013 from Kaggle (requires account + competition acceptance)...")
    try:
        od.download(
            "https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge",
            data_dir=str(data_dir),
        )
        # od creates subdir: data_dir/challenges-in-representation-learning-.../fer2013.csv
        for sub in data_dir.iterdir():
            if sub.is_dir():
                for f in sub.glob("*.csv"):
                    dest = fer_dir / f.name
                    shutil.copy(f, dest)
                    print(f"Copied {f.name} to {fer_dir}")
                for f in sub.glob("**/*.csv"):
                    if f.parent != sub:
                        continue
                    dest = fer_dir / f.name
                    if not dest.exists():
                        shutil.copy(f, dest)
                        print(f"Copied {f.name}")
    except Exception as e:
        print(f"Download failed: {e}")

    if not (fer_dir / "fer2013.csv").exists() and not (fer_dir / "icml_face_data.csv").exists():
        print("\nManual download:")
        print("1. Get fer2013.csv from Kaggle:")
        print("   https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
        print("2. Save to:", fer_dir.resolve())
        sys.exit(1)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
