"""
Video stream manager for multiple camera inputs.
Supports local files, webcams, RTSP, and multiple parallel streams.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

import cv2
import numpy as np


class VideoStreamManager:
    """
    Manages one or more video streams.
    Can run on single or multiple servers (each handles assigned streams).
    """

    def __init__(
        self,
        sources: List[Union[int, str, Path]],
        process_every_n: int = 1,
        max_queue_size: int = 2,
    ):
        """
        Args:
            sources: List of camera indices (0, 1...) or paths/URLs
            process_every_n: Process every N-th frame (1 = all)
            max_queue_size: Max frames to queue per source
        """
        self.sources = sources
        self.process_every_n = process_every_n
        self.max_queue_size = max_queue_size
        self._caps: List[cv2.VideoCapture] = []
        self._stop = False

    def open(self) -> bool:
        """Open all video sources."""
        for src in self.sources:
            cap = cv2.VideoCapture(int(src) if isinstance(src, int) else str(src))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {src}")
            self._caps.append(cap)
        return True

    def close(self):
        """Release all captures."""
        self._stop = True
        for cap in self._caps:
            cap.release()
        self._caps.clear()

    def read_frame(self, source_idx: int = 0) -> tuple[bool, Optional[np.ndarray], int]:
        """
        Read single frame from source.

        Returns:
            (success, frame, source_idx)
        """
        if source_idx >= len(self._caps):
            return False, None, source_idx

        ret, frame = self._caps[source_idx].read()
        return ret, frame if ret else None, source_idx

    def iter_frames(
        self,
        source_idx: Optional[int] = None,
        skip_duplicates: bool = True,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """
        Iterate over frames from one or all sources.

        Yields:
            (frame, source_idx)
        """
        indices = list(range(len(self._caps))) if source_idx is None else [source_idx]
        frame_counts = {i: 0 for i in indices}

        while not self._stop:
            for i in indices:
                ret, frame, _ = self.read_frame(i)
                if not ret or frame is None:
                    continue

                frame_counts[i] += 1
                if self.process_every_n > 1 and frame_counts[i] % self.process_every_n != 0:
                    continue

                yield frame, i

    def run_with_callback(
        self,
        callback: Callable[[np.ndarray, int], None],
        source_idx: Optional[int] = None,
    ):
        """Run stream and call callback for each frame."""
        for frame, idx in self.iter_frames(source_idx):
            callback(frame, idx)
            if self._stop:
                break
