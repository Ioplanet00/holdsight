"""FastAPI service. Uploaded video lives in a temp file only for the request and is deleted after."""
from __future__ import annotations
import os
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from .codes import Hold, WHY
from .config import Thresholds
from .face import FaceFinder
from .pipeline import analyse

MAX_BYTES = 60 * 1024 * 1024
T = Thresholds.load(os.environ.get("HOLDSIGHT_THRESHOLDS", "config/thresholds.yaml"))
app = FastAPI(title="HoldSight", version="0.1.0")
_finder: FaceFinder | None = None


def finder() -> FaceFinder:
    global _finder
    if _finder is None:
        _finder = FaceFinder(score_min=T.face_score_min)
    return _finder


@app.get("/v1/health")
def health():
    import cv2 as cv
    return {"ok": True, "opencv": cv.__version__}


@app.get("/v1/contract")
def contract():
    return {"states": ["value", "hold"], "codes": {c.value: WHY[c] for c in Hold},
            "note": "Screening signals only. Not a diagnosis, not a medical device, not vehicle control."}


@app.post("/v1/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    fd, path = tempfile.mkstemp(suffix=".mp4", dir=os.environ.get("HOLDSIGHT_TMP", "/dev/shm" if os.path.isdir("/dev/shm") else None))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        del data
        return analyse(path, T, finder())
    finally:
        os.remove(path)   # video is deleted right after processing


if os.path.isdir("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="web")
