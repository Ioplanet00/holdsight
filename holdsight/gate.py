"""Two-level quality gate: frame checks, then session checks. Fails produce hold codes, never values."""
from __future__ import annotations
from .codes import Hold, FRAME_PRECEDENCE
from .config import Thresholds
from .eye import EyeReading


def frame_hold(face_ok: bool, motion_iod: float | None, eye: EyeReading | None, t: Thresholds,
               face_mean: float | None = None, frame_mean: float | None = None) -> Hold | None:
    """Return the first failing check in FRAME_PRECEDENCE order, or None if the frame is usable.

    Darkness is judged on the whole face box, so a dark eye box on a well-lit face is treated as a
    covered eye (NO_EYE), not LOW_LIGHT. Blur on a dark frame is reported as LOW_LIGHT, because
    low light alone lowers the Laplacian variance."""
    failed: set[Hold] = set()
    if not face_ok:
        # a frame too dark for the detector is a light problem, not a missing person
        dark_frame = frame_mean is not None and frame_mean < t.low_light_face_mean_min
        failed.add(Hold.LOW_LIGHT if dark_frame else Hold.NO_FACE)
    else:
        dark = (face_mean is not None and face_mean < t.low_light_face_mean_min) or \
               (face_mean is None and eye is not None and eye.mean < t.low_light_mean_min)
        if dark:
            failed.add(Hold.LOW_LIGHT)
        if motion_iod is not None and motion_iod > t.motion_max_iod:
            failed.add(Hold.MOTION)
        if eye is None:
            failed.add(Hold.NO_EYE)
        else:
            if eye.glare_ratio > t.glare_ratio_max or eye.glare_blob > t.glare_blob_max:
                failed.add(Hold.GLARE)
            if eye.texture < t.occlusion_texture_min or eye.flat > t.occlusion_flat_max or (not dark and eye.mean < t.low_light_mean_min):
                failed.add(Hold.NO_EYE)
            if eye.blur_var < t.blur_var_min:
                failed.add(Hold.MOTION)
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
