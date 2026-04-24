"""Text emotion analysis using a local RuBERT sentiment model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional runtime dependency
    pipeline = None


class RuBertTextEmotionAnalyzer:
    """Infer coarse text emotion (positive/negative/neutral) from transcript text."""

    def __init__(self, model_dir: str) -> None:
        if pipeline is None:
            raise RuntimeError("Transformers is not installed. Run: pip install transformers")

        model_path = Path(model_dir)
        if not model_path.exists() or not model_path.is_dir():
            raise RuntimeError(f"RuBERT model directory not found: {model_dir}")

        self._classifier = pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            truncation=True,
            max_length=256,
        )

    def analyze(self, text: str) -> Dict[str, object]:
        clean = (text or "").strip()
        if not clean:
            return {
                "label": "neutral",
                "emotion": "NEUTRAL",
                "confidence": 0.0,
                "color": "#E2E8F0",
            }

        raw = self._classifier(clean)[0]
        raw_label = str(raw.get("label", "neutral")).strip().lower()
        score = float(raw.get("score", 0.0))
        normalized = self._normalize_label(raw_label)

        return {
            "label": normalized,
            "emotion": self._label_to_emotion(normalized),
            "confidence": score,
            "color": self._label_to_color(normalized),
        }

    def _normalize_label(self, label: str) -> str:
        known_positive = {"positive", "pos", "label_2", "2", "5", "4"}
        known_negative = {"negative", "neg", "label_0", "0", "1"}
        known_neutral = {"neutral", "neu", "label_1", "1", "3"}

        if label in known_positive:
            return "positive"
        if label in known_negative:
            return "negative"
        if label in known_neutral:
            return "neutral"

        if "pos" in label:
            return "positive"
        if "neg" in label:
            return "negative"
        return "neutral"

    def _label_to_emotion(self, label: str) -> str:
        if label == "positive":
            return "POSITIVE"
        if label == "negative":
            return "NEGATIVE"
        return "NEUTRAL"

    def _label_to_color(self, label: str) -> str:
        if label == "positive":
            return "#22C55E"
        if label == "negative":
            return "#FB7185"
        return "#7DD3FC"
