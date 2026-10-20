"""HoldSight pipeline: decode -> YuNet -> eye ROI -> CLAHE/threshold/contours -> gate -> driver profile."""
from __future__ import annotations
import time
import cv2 as cv
from .codes import Hold, WHY
from .config import Thresholds
from .decode import frames, delivered_fps
from .eye import crop, read_eye
from .face import FaceFinder
from .gate import frame_hold, session_hold
from .profiles import driver


def analyse(path: str, t: Thresholds | None = None, finder: FaceFinder | None = None, timing: bool = False,
            series: bool = False) -> dict:
    t = t or Thresholds()
    finder = finder or FaceFinder(score_min=t.face_score_min)
    ts_list, aperture, holds, signal = [], [], [], []
    stage_ms = {"detect": 0.0, "eye": 0.0}
    prev_centre = None
    for i, ts, bgr in frames(path):
        ts_list.append(ts)
        t0 = time.perf_counter()
        face = finder.find(bgr)
        t1 = time.perf_counter()
        eye = None
        motion = None
        face_mean = None
        frame_mean = None
        eye_sig = None
        if face is not None and face.iod > 0:
            if prev_centre is not None:
                dx, dy = face.centre[0] - prev_centre[0], face.centre[1] - prev_centre[1]
                motion = (dx * dx + dy * dy) ** 0.5 / face.iod
            prev_centre = face.centre
            gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
            fx, fy, fw, fh = (int(round(v)) for v in face.box)
            fbox = gray[max(0, fy):max(0, fy + fh), max(0, fx):max(0, fx + fw)]
            face_mean = float(fbox.mean()) if fbox.size else None
            side = t.eye_box_scale * face.iod * 2
            # mean of both eyes when both are readable; otherwise the readable one
            readings = [read_eye(e, t) for e in (crop(gray, face.right_eye, side), crop(gray, face.left_eye, side)) if e is not None]
            if readings:
                def bad(r):
                    return r.flat > t.occlusion_flat_max or r.glare_blob > t.glare_blob_max
                eye = min(readings, key=lambda r: (r.aperture is None, bad(r), r.glare_ratio))
                aps = [r.aperture for r in readings if r.aperture is not None and not bad(r)]
                if aps:
                    eye.aperture = sum(aps) / len(aps)
                good = [r for r in readings if not bad(r)] or [eye]
                eye_sig = (sum(r.dark_p5 for r in good) / len(good), sum(r.dark_share for r in good) / len(good))
        else:
            prev_centre = None
            frame_mean = float(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).mean())
        t2 = time.perf_counter()
        stage_ms["detect"] += (t1 - t0) * 1000
        stage_ms["eye"] += (t2 - t1) * 1000
        h = frame_hold(face is not None and face.score >= t.face_score_min, motion, eye, t, face_mean, frame_mean)
        holds.append(h)
        aperture.append(None if h is not None else eye.aperture)
        signal.append(None if h is not None else eye_sig)

    n = len(ts_list)
    fps = delivered_fps(ts_list)
    duration = (ts_list[-1] - ts_list[0]) / 1000.0 if n > 1 else 0.0
    usable_n = sum(1 for h in holds if h is None)
    usable_share = usable_n / n if n else 0.0
    usable_s = usable_n / fps if fps > 0 else 0.0
    s_hold = session_hold(fps, usable_share, duration, t)
    hold_counts = {c.value: sum(1 for h in holds if h == c) for c in Hold}
    out = {
        "frames": n, "delivered_fps": fps, "duration_s": duration, "usable_share": usable_share,
        "frame_holds": hold_counts,
        "timeline": [None if h is None else h.value for h in holds],
        "opencv": cv.__version__,
    }
    if s_hold is not None:
        out["session"] = {"hold": s_hold.value, "why": WHY[s_hold]}
    else:
        out["session"] = {"value": driver.summarise(signal, ts_list, usable_s, t)}
    if series:   # per-frame data for offline evaluation
        out["aperture"] = aperture
        out["signal"] = signal
        out["ts_ms"] = ts_list
    if timing and n:
        out["ms_per_frame"] = {k: v / n for k, v in stage_ms.items()}
    return out
