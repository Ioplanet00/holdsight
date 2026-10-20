"""Driver profile: blinks, blink rate, PERCLOS and longest closure from usable frames only.

Blink detector (chosen on Eyeblink8 videos 1-2, checked on 3-4, then frozen; see docs/EVALUATION.md):
  1. per usable frame, two eye-darkness signals: 5th-percentile grey level (rises when the lid covers
     the iris) and dark-pixel share (falls);
  2. each signal minus its rolling median over blink_window_s, divided by a rolling robust spread
     (1.4826 x median absolute deviation over 3 x blink_window_s) -> z-score; signals averaged;
  3. a blink is a run of frames with z > blink_z lasting 1 frame .. blink_max_s, starting at least
     blink_refractory_s after the previous event. Longer runs (up to closure_max_s) are closures.
Held frames carry no signal: they are bridged for the baseline only and can never be closed frames.
Returns times in ms.
"""
from __future__ import annotations
import numpy as np
from ..config import Thresholds


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, win | 1)
    pad = win // 2
    xp = np.pad(x, pad, mode="reflect") if len(x) > pad else np.pad(x, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(xp, win)
    return np.median(view, axis=1)


def blink_z(signal: list, ts_ms: list[float], t: Thresholds) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(signal)
    held = np.array([s is None for s in signal])
    if n == 0 or held.all():
        return np.zeros(n), held, 0.0
    span = (ts_ms[-1] - ts_ms[0]) / 1000.0 if n > 1 else 0.0
    fps = (n - 1) / span if span > 0 else 30.0
    win = int(round(t.blink_window_s * fps))
    zs = []
    for k, sign in ((0, 1.0), (1, -1.0)):
        x = np.array([np.nan if s is None else s[k] for s in signal], dtype=float)
        ok = ~held
        x[held] = np.interp(np.flatnonzero(held), np.flatnonzero(ok), x[ok]) if held.any() else x[held]
        d = sign * (x - _rolling_median(x, win))
        spread = _rolling_median(np.abs(d), 3 * win) * 1.4826 + 1e-6
        zs.append(d / spread)
    z = np.mean(zs, axis=0)
    z[held] = 0.0
    return z, held, fps


def blink_events(signal: list, ts_ms: list[float], t: Thresholds):
    """Returns (blinks, closures, fps) as lists of (start_ms, end_ms)."""
    z, held, fps = blink_z(signal, ts_ms, t)
    if fps == 0.0:
        return [], [], None
    max_blink = max(1, int(round(t.blink_max_s * fps)))
    max_closure = int(round(t.closure_max_s * fps))
    refractory = int(round(t.blink_refractory_s * fps))
    on = z > t.blink_z
    blinks, closures, last_end, i, n = [], [], -10 ** 9, 0, len(z)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        length = j - i
        if length >= t.blink_min_frames and i - last_end >= refractory:
            if length <= max_blink:
                blinks.append((ts_ms[i], ts_ms[j - 1]))
                last_end = j - 1
            elif length <= max_closure:
                closures.append((ts_ms[i], ts_ms[j - 1]))
                last_end = j - 1
        i = j
    return blinks, closures, fps


def summarise(signal, ts_ms, usable_s: float, t: Thresholds) -> dict:
    blinks, closures, fps = blink_events(signal, ts_ms, t)
    usable_ts = [ts for s, ts in zip(signal, ts_ms) if s is not None]
    closed = sum(1 for ts in usable_ts if any(a <= ts <= b for a, b in blinks + closures))
    perclos = closed / len(usable_ts) if usable_ts else None
    frame_s = 1.0 / fps if fps else 0.0
    longest = max([(e - s) / 1000.0 + frame_s for s, e in blinks + closures], default=0.0)
    return {
        "blink_count": len(blinks),
        "blink_rate_per_min": (len(blinks) / usable_s * 60.0) if usable_s > 0 else None,
        "perclos": perclos,
        "longest_closure_s": longest,
        "blinks_ms": blinks,
    }
