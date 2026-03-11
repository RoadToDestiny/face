#!/usr/bin/env python3
"""
Run emotion recognition on video file or image.
Usage:
  python scripts/run_inference.py video.mp4
  python scripts/run_inference.py image.jpg
  python scripts/run_inference.py 0  # webcam
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.pipeline import EmotionPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Video file, image file, or camera index (0)")
    ap.add_argument("--model", default=None, help="Path to emotion model checkpoint (omit to use untrained)")
    ap.add_argument("--output", default=None, help="Output video path (for video input)")
    ap.add_argument("--show", action="store_true", default=True, help="Show preview window")
    ap.add_argument("--no-show", action="store_false", dest="show")
    args = ap.parse_args()

    model_path = args.model
    if model_path and not Path(model_path).exists():
        print(f"Warning: Model {model_path} not found. Using untrained (ImageNet) weights.")
        model_path = None

    pipeline = EmotionPipeline(
        model_checkpoint=model_path,
        face_confidence=0.85,
        input_size=112,
    )

    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        # Try as image
        img = cv2.imread(args.input)
        if img is None:
            print(f"Cannot open: {args.input}")
            sys.exit(1)

        results = pipeline.process_frame(img)
        out = pipeline.draw_results(img, results)
        cv2.imshow("Emotion", out)
        print(f"Emotions: {[r['emotion'] for r in results]}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = pipeline.process_frame(frame)
            out = pipeline.draw_results(frame, results)

            if args.show:
                cv2.imshow("Emotion Recognition", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(out)
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
