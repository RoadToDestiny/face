#!/usr/bin/env python3
"""
Train emotion recognition model on FER2013.
Achieves ~70% on FER2013 (challenging dataset). For 95%+ consider RAF-DB or ensemble.
"""

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.emotion import create_emotion_model
from src.emotion.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data", help="FER2013 data directory")
    ap.add_argument("--output", default="./checkpoints", help="Checkpoint output dir")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--input-size", type=int, default=112)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    model = create_emotion_model(num_classes=7, backbone=args.backbone, pretrained=True)
    best_acc = train(
        model=model,
        data_dir=args.data_dir,
        output_dir=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        input_size=args.input_size,
        num_workers=args.workers,
    )
    print(f"\nBest test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
