"""Face and eye-centre landmarks with cv.FaceDetectorYN (YuNet, OpenCV Zoo, MIT licence)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2 as cv
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


@dataclass
class Face:
    box: tuple[float, float, float, float]   # x, y, w, h
    score: float
    right_eye: tuple[float, float]           # subject's right eye (image left)
    left_eye: tuple[float, float]

    @property
    def iod(self) -> float:
        return float(np.hypot(self.left_eye[0] - self.right_eye[0], self.left_eye[1] - self.right_eye[1]))

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2


class FaceFinder:
    def __init__(self, model: str | Path = DEFAULT_MODEL, score_min: float = 0.8):
        if not Path(model).exists():
            raise FileNotFoundError(f"{model} missing - run `make models`")
        self.det = cv.FaceDetectorYN.create(str(model), "", (320, 320), score_min, 0.3, 5000)
        self.size = None

    def find(self, bgr: np.ndarray) -> Face | None:
        h, w = bgr.shape[:2]
        if self.size != (w, h):
            self.det.setInputSize((w, h))
            self.size = (w, h)
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])   # largest face
        return Face(box=tuple(map(float, f[0:4])), score=float(f[14]),
                    right_eye=(float(f[4]), float(f[5])), left_eye=(float(f[6]), float(f[7])))
