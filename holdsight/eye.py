"""Eye-region measures with classic OpenCV 5 operations.

crop -> cv.resize -> cv.createCLAHE -> cv.adaptiveThreshold -> cv.morphologyEx
-> cv.findContours -> aperture (h/w of dominant dark contour) + cv.moments pupil centre.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2 as cv
import numpy as np
from .config import Thresholds


@dataclass
class EyeReading:
    aperture: float | None      # height/width of dominant dark blob; None if no valid contour
    pupil: tuple[float, float] | None   # normalised (0..1) within eye box
    mean: float                 # eye-box mean intensity after CLAHE
    glare_ratio: float
    blur_var: float


def crop(gray: np.ndarray, centre: tuple[float, float], side: float) -> np.ndarray | None:
    cx, cy = centre
    half = side / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half * 0.7))
    x1, y1 = int(round(cx + half)), int(round(cy + half * 0.7))
    H, W = gray.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 - x0 < 8 or y1 - y0 < 6:
        return None
    return gray[y0:y1, x0:x1]


def read_eye(eye_gray: np.ndarray, t: Thresholds) -> EyeReading:
    raw = cv.resize(eye_gray, (t.eye_resize, int(t.eye_resize * 0.7)), interpolation=cv.INTER_AREA)
    glare_ratio = float(np.count_nonzero(raw >= t.glare_level)) / raw.size
    blur_var = float(cv.Laplacian(raw, cv.CV_64F).var())
    clahe = cv.createCLAHE(clipLimit=t.clahe_clip, tileGridSize=(t.clahe_tile, t.clahe_tile))
    eq = clahe.apply(raw)
    mean = float(raw.mean())
    blur = cv.GaussianBlur(eq, (5, 5), 0)
    mask = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
                                t.adapt_block, t.adapt_c)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= t.min_eye_contour_area]
    if not contours:
        return EyeReading(None, None, mean, glare_ratio, blur_var)
    # dominant contour = largest area near the box centre
    h, w = mask.shape
    def score(c):
        x, y, cw, ch = cv.boundingRect(c)
        dx = (x + cw / 2 - w / 2) / w
        dy = (y + ch / 2 - h / 2) / h
        return cv.contourArea(c) * (1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5))
    c = max(contours, key=score)
    x, y, cw, ch = cv.boundingRect(c)
    aperture = ch / cw if cw > 0 else None
    m = cv.moments(c)
    pupil = (m["m10"] / m["m00"] / w, m["m01"] / m["m00"] / h) if m["m00"] > 0 else None
    return EyeReading(aperture, pupil, mean, glare_ratio, blur_var)
