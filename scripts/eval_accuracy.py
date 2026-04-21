#!/usr/bin/env python3
"""
Evaluate emotion model accuracy on FER2013 test set.
Run after training to measure achieved metrics.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from src.emotion.fer2013_dataset import get_fer2013_dataset
from torchvision import transforms
from tqdm import tqdm

from src.emotion import create_emotion_model


class LabelMapTransform:
    """Picklable target transform for Windows multiprocessing."""

    def __init__(self, label_map):
        self.label_map = [int(v) for v in label_map]

    def __call__(self, y):
        return self.label_map[y]


def get_transforms(input_size=112):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--input-size", type=int, default=112)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_emotion_model(checkpoint=args.checkpoint)
    model.to(device)
    model.eval()

    data_path = Path(args.data_dir)
    test_path = data_path / "test"
    FER2013_ORDER = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    if test_path.exists():
        from torchvision.datasets import ImageFolder
        from src.emotion.trainer import get_fer2013_transforms
        class_names = sorted(p.name for p in test_path.iterdir() if p.is_dir())
        label_map = [FER2013_ORDER.index(c) for c in class_names]
        target_transform = LabelMapTransform(label_map)

        ds = ImageFolder(
            test_path,
            transform=get_fer2013_transforms(train=False, input_size=args.input_size),
            target_transform=target_transform,
        )
        labels = FER2013_ORDER
    else:
        ds = get_fer2013_dataset(split="test", data_dir=args.data_dir, transform=get_transforms(args.input_size))
        labels = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    num_classes = len(labels)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            preds = model(images).argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

            for t, p in zip(targets.cpu(), preds.cpu()):
                t, p = int(t), int(p)
                if t < num_classes:
                    class_total[t] += 1
                    if t == p:
                        class_correct[t] += 1

    acc = correct / total if total > 0 else 0
    print(f"\nOverall accuracy: {acc:.4f} ({correct}/{total})")
    print("\nPer-class accuracy:")
    for i, name in enumerate(labels):
        a = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {name}: {a:.4f} ({class_correct[i]}/{class_total[i]})")


if __name__ == "__main__":
    main()
