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
            "anger": (113, 113, 248),
            "disgust": (53, 229, 163),
            "fear": (253, 181, 196),
            "happy": (172, 239, 134),
            "sad": (253, 197, 147),
            "surprise": (77, 211, 252),
            "neutral": (235, 231, 229),
        }

        def draw_rounded_rect(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], line_w: int, radius: int) -> None:
            x1, y1 = p1
            x2, y2 = p2
            line_w = max(1, line_w)
            radius = max(2, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, line_w, cv2.LINE_AA)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, line_w, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, line_w, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, line_w, cv2.LINE_AA)

            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, line_w, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, line_w, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, line_w, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, line_w, cv2.LINE_AA)

        def fill_rounded_rect(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], radius: int) -> None:
            x1, y1 = p1
            x2, y2 = p2
            radius = max(2, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            emotion = r["emotion"]
            conf = r["confidence"]
            color = colors.get(emotion, (255, 255, 255))

            h, w = out.shape[:2]
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w - 1, int(x2)))
            y2 = max(0, min(h - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            radius = max(8, min(18, int(min(x2 - x1, y2 - y1) * 0.08)))

            # Layered glow to match glass UI aesthetics.
            for glow_w, alpha in ((12, 0.08), (8, 0.12), (5, 0.17)):
                glow = out.copy()
                draw_rounded_rect(glow, (x1, y1), (x2, y2), color, glow_w, radius)
                out = cv2.addWeighted(glow, alpha, out, 1.0 - alpha, 0)

            draw_rounded_rect(out, (x1, y1), (x2, y2), color, max(2, thickness), radius)

            label = f"{emotion.upper()}  {int(conf * 100)}%"
            text_w, text_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            pad_x, pad_y = 12, 8
            panel_w = text_w + pad_x * 2
            panel_h = text_h + pad_y * 2

            panel_x1 = max(6, min(w - panel_w - 6, x1))
            panel_y2 = y1 - 8
            panel_y1 = panel_y2 - panel_h
            if panel_y1 < 6:
                panel_y1 = min(h - panel_h - 6, y2 + 8)
                panel_y2 = panel_y1 + panel_h

            panel = out.copy()
            fill_rounded_rect(panel, (panel_x1, panel_y1), (panel_x1 + panel_w, panel_y2), (18, 24, 36), 10)
            out = cv2.addWeighted(panel, 0.55, out, 0.45, 0)
            draw_rounded_rect(out, (panel_x1, panel_y1), (panel_x1 + panel_w, panel_y2), color, 1, 10)

            text_x = panel_x1 + pad_x
            text_y = panel_y1 + panel_h - pad_y
            cv2.putText(out, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (248, 250, 252), 1, cv2.LINE_AA)

        return out
