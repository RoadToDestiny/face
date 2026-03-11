"""
FER2013 dataset loader.
Uses torchvision FER2013 - requires fer2013.csv in data_dir/fer2013/.
Run: python scripts/download_fer2013.py --data-dir ./data
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset


def get_fer2013_dataset(
    split: str,
    data_dir: str,
    transform: Optional[Callable] = None,
) -> Dataset:
    """
    Load FER2013 via torchvision. Requires fer2013.csv in data_dir/fer2013/.
    Use scripts/download_fer2013.py to download.
    """
    from torchvision.datasets import FER2013
    return FER2013(root=data_dir, split=split, transform=transform)
