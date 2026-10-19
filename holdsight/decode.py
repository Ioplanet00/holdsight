"""Video decoding with OpenCV 5 (cv.VideoCapture)."""
from __future__ import annotations
from typing import Iterator, Tuple
import cv2 as cv
import numpy as np


def frames(path: str) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (index, timestamp_ms, BGR frame). Timestamps come from CAP_PROP_POS_MSEC,
    falling back to index / nominal fps when the container does not provide them."""
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    nominal = cap.get(cv.CAP_PROP_FPS) or 30.0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = cap.get(cv.CAP_PROP_POS_MSEC)
            if not ts and i > 0:
                ts = i * 1000.0 / nominal
            yield i, float(ts), frame
            i += 1
    finally:
        cap.release()


def delivered_fps(ts_ms: list[float]) -> float:
    if len(ts_ms) < 2:
        return 0.0
    span = ts_ms[-1] - ts_ms[0]
    return (len(ts_ms) - 1) * 1000.0 / span if span > 0 else 0.0
