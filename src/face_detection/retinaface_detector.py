"""
Face detection with MTCNN (PyTorch) or RetinaFace.
MTCNN is used by default - PyTorch native, no TensorFlow.
Provides bounding boxes and landmarks for downstream emotion recognition.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


def _get_mtcnn():
    """Lazy import to avoid loading model at module import."""
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=True, device="cuda" if __import__("torch").cuda.is_available() else "cpu")


class RetinaFaceDetector:
    """
    Face detector with alignment support.
    Uses MTCNN (PyTorch) for compatibility - no TensorFlow required.
    Alignment improves emotion recognition by normalizing face orientation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        align: bool = True,
        detection_size: Optional[Tuple[int, int]] = None,
        crop_padding_ratio: float = 0.12,
    ):
        self.confidence_threshold = confidence_threshold
        self.align = align
        self.detection_size = detection_size or (640, 640)
        self.crop_padding_ratio = max(0.0, min(0.35, float(crop_padding_ratio)))
        self._mtcnn = None

    def _ensure_mtcnn(self):
        if self._mtcnn is None:
            self._mtcnn = _get_mtcnn()

    def detect(
        self,
        image: np.ndarray,
        return_landmarks: bool = True,
    ) -> List[dict]:
        """
        Detect faces in BGR image.

        Returns:
            List of dicts with keys: bbox (x1,y1,x2,y2), confidence, landmarks, aligned_face
        """
        self._ensure_mtcnn()

        # MTCNN expects RGB, PIL or tensor
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # PIL for facenet
        from PIL import Image
        pil_img = Image.fromarray(image_rgb)

        boxes, probs, landmarks = self._mtcnn.detect(pil_img, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return []

        results = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None or prob < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            bbox = (x1, y1, x2, y2)

            result = {
                "bbox": bbox,
                "confidence": float(prob),
            }

            lm = landmarks[i] if landmarks is not None and i < len(landmarks) else None
            if return_landmarks and lm is not None and len(lm) >= 5:
                # MTCNN: left_eye, right_eye, nose, left_mouth, right_mouth
                result["landmarks"] = {
                    "left_eye": lm[0].tolist(),
                    "right_eye": lm[1].tolist(),
                    "nose": lm[2].tolist() if len(lm) > 2 else [0, 0],
                }

            if self.align and lm is not None and len(lm) >= 2:
                landmarks_dict = {
                    "left_eye": lm[0],
                    "right_eye": lm[1],
                }
                aligned = self._align_face(image_rgb, bbox, landmarks_dict)
                result["aligned_face"] = aligned

            results.append(result)

        return results

    def _align_face(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        landmarks: dict,
    ) -> np.ndarray:
        """Align face using eye positions."""
        x1, y1, x2, y2 = bbox
        left_eye = np.array(landmarks.get("left_eye", [x1, y1]))
        right_eye = np.array(landmarks.get("right_eye", [x2, y1]))

        w = x2 - x1
        h = y2 - y1
        pad = int(self.crop_padding_ratio * max(w, h))
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(image.shape[1], x2 + pad)
        y2_p = min(image.shape[0], y2 + pad)

        face_crop = image[y1_p:y2_p, x1_p:x2_p].copy()

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        if abs(angle) > 1.0:
            center = ((x2_p - x1_p) // 2, (y2_p - y1_p) // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            face_crop = cv2.warpAffine(face_crop, M, (x2_p - x1_p, y2_p - y1_p))

        return face_crop
