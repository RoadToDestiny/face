"""
End-to-end emotion recognition pipeline.
Face detection -> Crop -> Preprocess -> Emotion CNN -> Prediction
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from ..emotion import EmotionPreprocessor, create_emotion_model
from ..face_detection import RetinaFaceDetector


class EmotionPipeline:
    """
    Full pipeline: RetinaFace -> Preprocess -> ResNet Emotion.
    Optimized for real-time inference.
    """

    LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(
        self,
        model_checkpoint: Optional[Union[str, Path]] = None,
        face_confidence: float = 0.9,
        input_size: int = 112,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.detector = RetinaFaceDetector(
            confidence_threshold=face_confidence,
            align=True,
        )
        self.preprocessor = EmotionPreprocessor(
            input_size=input_size,
            grayscale=True,
            hist_eq=True,
            normalize=True,
        )
        self.model = create_emotion_model(
            num_classes=7,
            backbone="resnet18",
            pretrained=model_checkpoint is None,
            checkpoint=model_checkpoint,
        )
        self.model.to(self.device)
        self.model.eval()

        self.input_size = input_size

    def process_frame(
        self,
        frame: np.ndarray,
        detect_faces: bool = True,
    ) -> List[Dict]:
        """
        Process single frame. Detect faces, predict emotions.

        Returns:
            List of dicts: bbox, emotion, confidence, landmarks
        """
        if detect_faces:
            detections = self.detector.detect(frame, return_landmarks=True)
        else:
            # Assume full frame is a face (for cropped input)
            h, w = frame.shape[:2]
            detections = [{
                "bbox": (0, 0, w, h),
                "confidence": 1.0,
                "aligned_face": frame,
            }]

        results = []
        for det in detections:
            face_img = det.get("aligned_face")
            if face_img is None:
                x1, y1, x2, y2 = det["bbox"]
                face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            emotion, conf = self._predict_emotion(face_img)
            results.append({
                "bbox": det["bbox"],
                "emotion": emotion,
                "confidence": float(conf),
                "landmarks": det.get("landmarks"),
            })

        return results

    def _predict_emotion(self, face: np.ndarray) -> Tuple[str, float]:
        """Predict emotion for single face crop."""
        preprocessed = self.preprocessor.preprocess(face)
        x = torch.from_numpy(preprocessed).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(x)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return self.LABELS[idx], conf

    def draw_results(
        self,
        frame: np.ndarray,
        results: List[Dict],
        font_scale: float = 0.6,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and emotion labels on frame."""
        out = frame.copy()
        colors = {
            "anger": (0, 0, 255),
            "disgust": (0, 128, 128),
            "fear": (128, 0, 128),
            "happy": (0, 255, 0),
            "sad": (255, 0, 0),
            "surprise": (0, 255, 255),
            "neutral": (200, 200, 200),
        }

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            emotion = r["emotion"]
            conf = r["confidence"]
            color = colors.get(emotion, (255, 255, 255))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"{emotion} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        return out
