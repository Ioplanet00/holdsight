"""Two-level quality gate: frame checks, then session checks. Fails produce hold codes, never values."""
from __future__ import annotations
from .codes import Hold, FRAME_PRECEDENCE
from .config import Thresholds
from .eye import EyeReading


def frame_hold(face_ok: bool, motion_iod: float | None, eye: EyeReading | None, t: Thresholds) -> Hold | None:
    """Return the first failing check in FRAME_PRECEDENCE order, or None if the frame is usable."""
    failed: set[Hold] = set()
    if not face_ok:
        failed.add(Hold.NO_FACE)
    else:
        if motion_iod is not None and motion_iod > t.motion_max_iod:
            failed.add(Hold.MOTION)
        if eye is None:
            failed.add(Hold.NO_EYE)
        else:
            if eye.blur_var < t.blur_var_min:
                failed.add(Hold.MOTION)
            if eye.mean < t.low_light_mean_min:
                failed.add(Hold.LOW_LIGHT)
            if eye.glare_ratio > t.glare_ratio_max:
                failed.add(Hold.GLARE)
            if eye.aperture is None:
                failed.add(Hold.NO_EYE)
    for code in FRAME_PRECEDENCE:
        if code in failed:
            return code
    return None


def session_hold(fps: float, usable_share: float, duration_s: float, t: Thresholds) -> Hold | None:
    if fps < t.min_fps:
        return Hold.LOW_FPS
    if usable_share < t.min_usable_share or duration_s < t.min_duration_s:
        return Hold.LOW_USABLE
    return None
