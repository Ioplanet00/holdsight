"""FastAPI service. Uploaded video lives in a temp file only for the request and is deleted after."""
from __future__ import annotations
import os
import tempfile
import threading
import time
from collections import deque
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from .codes import Hold, WHY
from .config import Thresholds
from .face import FaceFinder
from .pipeline import analyse

MAX_BYTES = int(os.environ.get("HOLDSIGHT_MAX_BYTES", 30 * 1024 * 1024))
ALLOWED_EXT = {".mp4", ".mov", ".webm", ".avi", ".m4v"}
ALLOWED_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo", "video/avi", "application/octet-stream", ""}
RATE_PER_MIN = int(os.environ.get("HOLDSIGHT_RATE_PER_MIN", 6))
_hits: dict[str, deque] = {}
_hits_lock = threading.Lock()
_busy = threading.BoundedSemaphore(int(os.environ.get("HOLDSIGHT_MAX_CONCURRENT", 2)))
T = Thresholds.load(os.environ.get("HOLDSIGHT_THRESHOLDS", "config/thresholds.yaml"))
app = FastAPI(title="HoldSight", version="0.1.0")
_finder: FaceFinder | None = None


def finder() -> FaceFinder:
    global _finder
    if _finder is None:
        _finder = FaceFinder(score_min=T.face_score_min)
    return _finder


def _client(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?")


def _allow(ip: str) -> bool:
    now = time.monotonic()
    with _hits_lock:
        q = _hits.setdefault(ip, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            return False
        q.append(now)
        if len(_hits) > 10000:
            _hits.clear()
        return True


def _build_info() -> dict:
    import platform
    import cv2 as cv
    info = cv.getBuildInformation()
    keep = ("Version control", "Platform", "Host", "CPU/HW features", "Baseline", "Dispatched code",
            "Parallel framework", "Other third-party", "Lapack", "Eigen", "Custom HAL", "KleidiCV", "Extra dependencies")
    lines = [l.strip() for l in info.splitlines() if any(k in l for k in keep)]
    return {"opencv": cv.__version__, "module": cv.__file__, "threads": cv.getNumThreads(),
            "machine": platform.machine(), "build": lines[:40]}


@app.get("/v1/health")
def health():
    import cv2 as cv
    return {"ok": True, "opencv": cv.__version__}


@app.get("/v1/build")
def build():
    return _build_info()


@app.get("/v1/contract")
def contract():
    return {"states": ["value", "hold"], "codes": {c.value: WHY[c] for c in Hold},
            "note": "Screening signals only. Not a diagnosis, not a medical device, not vehicle control."}


@app.post("/v1/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    if not _allow(_client(request)):
        raise HTTPException(429, "too many requests, try again in a minute")
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BYTES + 1024 * 1024:
        raise HTTPException(413, "file too large")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT or (file.content_type or "") not in ALLOWED_TYPES:
        raise HTTPException(415, "unsupported file type")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    if not _busy.acquire(blocking=False):
        raise HTTPException(503, "server busy, try again shortly")
    fd, path = tempfile.mkstemp(suffix=".mp4", dir=os.environ.get("HOLDSIGHT_TMP", "/dev/shm" if os.path.isdir("/dev/shm") else None))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        del data
        t0 = time.perf_counter()
        res = analyse(path, T, finder())
        ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(res, dict):
            import cv2 as cv
            n = len(res.get("timeline") or [])
            res["perf"] = {"total_ms": round(ms, 1), "frames": n,
                           "ms_per_frame": round(ms / n, 2) if n else None, "opencv": cv.__version__}
        return res
    finally:
        _busy.release()
        os.remove(path)   # video is deleted right after processing


if os.path.isdir("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="web")
