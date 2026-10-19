"""Driver profile: blinks, blink rate, PERCLOS and longest closure from usable frames only."""
from __future__ import annotations
import numpy as np
from ..config import Thresholds


def blink_events(aperture: list[float | None], ts_ms: list[float], t: Thresholds):
    """Hysteresis on aperture normalised by the session open-eye baseline (90th percentile of
    usable frames). Held frames (None) end any open event and are never counted as closed.
    Returns (blinks, closures) as lists of (start_ms, end_ms)."""
    vals = np.array([a for a in aperture if a is not None], dtype=float)
    if vals.size == 0:
        return [], [], None
    base = float(np.percentile(vals, 90))
    if base <= 0:
        return [], [], None
    blinks, closures = [], []
    closed, start, n = False, None, 0
    for a, ts in zip(aperture, ts_ms):
        if a is None:
            closed, start, n = False, None, 0
            continue
        r = a / base
        if not closed and r < t.closed_ratio:
            closed, start, n = True, ts, 1
        elif closed and r < t.open_ratio:
            n += 1
        elif closed and r >= t.open_ratio:
            if n >= t.blink_min_frames:
                (blinks if (ts - start) / 1000.0 <= t.blink_max_s else closures).append((start, ts))
            closed, start, n = False, None, 0
    return blinks, closures, base


def summarise(aperture, ts_ms, usable_s: float, t: Thresholds) -> dict:
    blinks, closures, base = blink_events(aperture, ts_ms, t)
    usable = [a for a in aperture if a is not None]
    perclos = (float(np.mean([a / base < t.closed_ratio for a in usable])) if usable and base else None)
    longest = max([(e - s) / 1000.0 for s, e in blinks + closures], default=0.0)
    return {
        "blink_count": len(blinks),
        "blink_rate_per_min": (len(blinks) / usable_s * 60.0) if usable_s > 0 else None,
        "perclos": perclos,
        "longest_closure_s": longest,
        "blinks_ms": blinks,
    }
