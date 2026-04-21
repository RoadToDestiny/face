#!/usr/bin/env python3
"""
Train/fine-tune emotion model on custom dataset (e.g. Russian content).
Expects ImageFolder structure: data_dir/train/anger/, train/happy/, etc.
Use --checkpoint to fine-tune from pretrained model.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.emotion import create_emotion_model
from src.emotion.trainer import train_epoch, get_fer2013_transforms

# Порядок классов FER2013 (должен совпадать с EmotionPipeline.LABELS)
FER2013_ORDER = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class LabelMapTransform:
    """Picklable target transform for Windows multiprocessing."""

    def __init__(self, label_map):
        self.label_map = [int(v) for v in label_map]

    def __call__(self, y):
        return self.label_map[y]


def _build_label_map(class_names):
    """Map ImageFolder (alphabetical) indices to FER2013 indices."""
    return [FER2013_ORDER.index(c) for c in sorted(class_names)]


def train(
    model,
    data_dir,
    output_dir,
    checkpoint=None,
    batch_size=32,
    epochs=30,
    lr=0.0001,
    input_size=112,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_path = Path(data_dir) / "train"
    class_names = sorted(p.name for p in train_path.iterdir() if p.is_dir())
    label_map = _build_label_map(class_names)
    target_transform = LabelMapTransform(label_map)

    train_ds = ImageFolder(
        train_path,
        transform=get_fer2013_transforms(train=True, input_size=input_size),
        target_transform=target_transform,
    )
    test_path = Path(data_dir) / "test"
    test_ds = ImageFolder(
        test_path,
        transform=get_fer2013_transforms(train=False, input_size=input_size),
        target_transform=target_transform,
    ) if test_path.exists() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    ) if test_ds else None

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_acc = 0.0
        if test_loader:
            from src.emotion.trainer import evaluate
            _, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

        if test_loader and test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "accuracy": test_acc,
                    "input_size": input_size,
                    "classes": train_ds.classes,
                },
                Path(output_dir) / "best_emotion_model.pt",
            )
            print(f"  -> Saved (acc={test_acc:.4f})")
        elif not test_loader and train_acc > best_acc:
            best_acc = train_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "accuracy": train_acc,
                    "input_size": input_size,
                    "classes": train_ds.classes,
                },
                Path(output_dir) / "best_emotion_model.pt",
            )
            print(f"  -> Saved (train_acc={train_acc:.4f})")

    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to custom dataset (train/test folders)")
    ap.add_argument("--checkpoint", default=None, help="Path to pretrained model for fine-tuning")
    ap.add_argument("--output", default="./checkpoints", help="Output directory")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--input-size", type=int, default=112)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    model = create_emotion_model(
        num_classes=7,
        backbone=args.backbone,
        pretrained=args.checkpoint is None,
        checkpoint=args.checkpoint,
    )
    best_acc = train(
        model=model,
        data_dir=args.data_dir,
        output_dir=args.output,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        input_size=args.input_size,
        num_workers=args.workers,
    )
    print(f"\nBest accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
