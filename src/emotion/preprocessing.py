"""
Preprocessing for emotion recognition.
Optimized for FER-style input: grayscale, normalized, consistent size.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np


class EmotionPreprocessor:
    """
    Preprocess face crops for emotion CNN.
    Applies: resize, grayscale (optional), histogram equalization, normalization.
    """

    EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(
        self,
        input_size: int = 112,
        grayscale: bool = True,
        hist_eq: bool = True,
        normalize: bool = True,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
    ):
        self.input_size = input_size
        self.grayscale = grayscale
        self.hist_eq = hist_eq
        self.normalize = normalize
        # ImageNet stats for ResNet pretrained weights
        self.mean = mean or (0.485, 0.456, 0.406)
        self.std = std or (0.229, 0.224, 0.225)

    def __call__(self, face: np.ndarray) -> np.ndarray:
        """Preprocess single face image for inference."""
        return self.preprocess(face)

    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face for emotion model.

        Args:
            face: BGR or RGB image, any size

        Returns:
            Preprocessed array ready for model input (C, H, W), float32
        """
        # Ensure RGB if color
        if len(face.shape) == 2:
            img = face
        elif face.shape[2] == 3:
            img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if self.grayscale else cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        else:
            img = face[:, :, 0] if self.grayscale else face

        # Resize
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        # Histogram equalization (improves contrast for subtle expressions)
        if self.hist_eq and self.grayscale:
            img = cv2.equalizeHist(img)

        # To float
        img = img.astype(np.float32) / 255.0

        if self.grayscale:
            # ResNet expects 3 channels - replicate for pretrained weights
            img = np.stack([img, img, img], axis=0)  # (3, H, W)
        else:
            img = np.transpose(img, (2, 0, 1))  # (C, H, W)

        if self.normalize:
            mean = np.array(self.mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(self.std, dtype=np.float32).reshape(-1, 1, 1)
            img = (img - mean) / std

        return img.astype(np.float32)

    def preprocess_batch(self, faces: list[np.ndarray]) -> np.ndarray:
        """Preprocess batch of faces."""
        return np.stack([self.preprocess(f) for f in faces], axis=0)
