"""COOL vs pip OpenCV 5 benchmark (template).

Run the SAME script, SAME inputs, on the SAME instance type twice: once with the COOL build
(COOL AMI's OpenCV) and once inside a venv with opencv-python-headless==5.0.0.93.
Measures per-stage and per-frame latency (median, p95) and throughput; cost per 1,000 frames is
computed from an hourly price you pass in (on-demand EC2 + COOL software fee at measurement time).
Writes results/bench_<label>.json. No numbers are pre-filled.

  python -m eval.bench --label cool --clips data/bench/*.mp4 --repeat 3 --usd-per-hour <price>
"""
import argparse, glob, json, platform, statistics, time
from pathlib import Path
import cv2 as cv
import numpy as np
from holdsight.config import Thresholds
from holdsight.eye import read_eye

OPS = ["resize", "clahe", "adaptiveThreshold", "findContours", "yunet"]


def pct(v, q):
    return float(np.percentile(np.array(v), q)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--usd-per-hour", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0, help="cv.setNumThreads; 0 = OpenCV default")
    a = ap.parse_args()
    if a.threads:
        cv.setNumThreads(a.threads)
    from holdsight.face import FaceFinder
    t = Thresholds.load("config/thresholds.yaml")
    finder = FaceFinder(score_min=t.face_score_min)
    clips = sorted(sum((glob.glob(c) for c in a.clips), []))
    per_frame, per_op = [], {k: [] for k in OPS}
    n_frames = 0
    wall0 = time.perf_counter()
    for _ in range(a.repeat):
        for clip in clips:
            cap = cv.VideoCapture(clip)
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                f0 = time.perf_counter()
                s = time.perf_counter(); face = finder.find(bgr); per_op["yunet"].append((time.perf_counter() - s) * 1e3)
                gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
                h, w = gray.shape
                roi = gray[h // 4: h // 2, w // 4: 3 * w // 4] if face is None else gray
                s = time.perf_counter(); r = cv.resize(roi, (t.eye_resize, int(t.eye_resize * 0.7))); per_op["resize"].append((time.perf_counter() - s) * 1e3)
                s = time.perf_counter(); e = cv.createCLAHE(t.clahe_clip, (t.clahe_tile, t.clahe_tile)).apply(r); per_op["clahe"].append((time.perf_counter() - s) * 1e3)
                s = time.perf_counter(); m = cv.adaptiveThreshold(e, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, t.adapt_block, t.adapt_c); per_op["adaptiveThreshold"].append((time.perf_counter() - s) * 1e3)
                s = time.perf_counter(); cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE); per_op["findContours"].append((time.perf_counter() - s) * 1e3)
                per_frame.append((time.perf_counter() - f0) * 1e3)
                n_frames += 1
            cap.release()
    wall = time.perf_counter() - wall0
    fps = n_frames / wall if wall > 0 else None
    res = {
        "label": a.label, "opencv": cv.__version__, "machine": platform.machine(),
        "threads": cv.getNumThreads(), "clips": clips, "repeat": a.repeat, "frames": n_frames,
        "frame_ms": {"median": pct(per_frame, 50), "p95": pct(per_frame, 95)},
        "op_ms": {k: {"median": pct(v, 50), "p95": pct(v, 95)} for k, v in per_op.items()},
        "throughput_fps": fps,
        "usd_per_1000_frames": (a.usd_per_hour / 3600.0 * 1000.0 / fps) if (a.usd_per_hour and fps) else None,
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/bench_{a.label}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
