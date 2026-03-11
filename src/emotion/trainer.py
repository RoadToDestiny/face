"""
Training pipeline for emotion recognition model.
Uses FER2013 dataset with strong augmentation for robust recognition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from .fer2013_dataset import get_fer2013_dataset
from tqdm import tqdm


def get_fer2013_transforms(train: bool, input_size: int = 112):
    """Transforms for FER2013 - aligned with inference preprocessing."""
    base = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if train:
        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
            ], p=0.3),
        ]
        return transforms.Compose(augment + base)

    return transforms.Compose(base)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train(
    model: nn.Module,
    data_dir: str,
    output_dir: str = "./checkpoints",
    batch_size: int = 64,
    epochs: int = 80,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    input_size: int = 112,
    num_workers: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    fer_dir = Path(data_dir) / "fer2013"
    if not (fer_dir / "fer2013.csv").exists() and not (fer_dir / "icml_face_data.csv").exists():
        raise FileNotFoundError(
            f"FER2013 data not found in {fer_dir}. "
            "Run: python scripts/download_fer2013.py --data-dir " + str(Path(data_dir).resolve()) + "\n"
            "Or download fer2013.csv from Kaggle and place in that directory."
        )

    train_ds = get_fer2013_dataset(
        split="train",
        data_dir=data_dir,
        transform=get_fer2013_transforms(train=True, input_size=input_size),
    )
    test_ds = get_fer2013_dataset(
        split="test",
        data_dir=data_dir,
        transform=get_fer2013_transforms(train=False, input_size=input_size),
    )

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
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "accuracy": test_acc,
                "input_size": input_size,
            }, Path(output_dir) / "best_emotion_model.pt")
            print(f"  -> Saved best model (acc={test_acc:.4f})")

    return best_acc
