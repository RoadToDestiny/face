#!/usr/bin/env python3
"""
Real-time emotion recognition from multiple cameras.
Usage:
  python scripts/run_stream.py 0           # single webcam
  python scripts/run_stream.py 0 1 2      # multiple cameras
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.pipeline import EmotionPipeline
from src.streams import VideoStreamManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="Camera indices or video URLs")
    ap.add_argument("--model", default=None, help="Path to emotion model checkpoint")
    ap.add_argument("--skip", type=int, default=1, help="Process every N-th frame")
    args = ap.parse_args()

    sources = [int(s) if s.isdigit() else s for s in args.sources]

    pipeline = EmotionPipeline(
        model_checkpoint=args.model,
        face_confidence=0.85,
        input_size=112,
    )

    manager = VideoStreamManager(
        sources=sources,
        process_every_n=args.skip,
        max_queue_size=2,
    )
    manager.open()

    def process(frame, idx):
        results = pipeline.process_frame(frame)
        out = pipeline.draw_results(frame, results)
        cv2.imshow(f"Camera {idx}", out)

    try:
        for frame, idx in manager.iter_frames():
            process(frame, idx)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        manager.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
