"""Video decoding with OpenCV 5 (cv.VideoCapture)."""
from __future__ import annotations
import os
import queue
import threading
from typing import Iterator, Tuple
import cv2 as cv
import numpy as np


def frames(path: str, prefetch: int | None = None) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Decode on a background thread so decoding overlaps detection (HOLDSIGHT_PREFETCH frames
    buffered, default 16; 0 = decode inline)."""
    n = int(os.environ.get("HOLDSIGHT_PREFETCH", 16)) if prefetch is None else prefetch
    if n <= 0:
        yield from _frames(path)
        return
    q: queue.Queue = queue.Queue(maxsize=n)
    stop = threading.Event()
    err: list[BaseException] = []

    def work():
        try:
            for item in _frames(path):
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as e:   # surfaced in the consumer
            err.append(e)
        finally:
            q.put(None)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        if err:
            raise err[0]
    finally:
        stop.set()
        while th.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                th.join(0.05)


def _frames(path: str) -> Iterator[Tuple[int, float, np.ndarray]]:
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
