"""Controlled degradations for hold-correctness evaluation (template).

From a clean consented clip, write degraded copies whose degraded frame ranges are known:
dark (brightness x0.15), blur (Gaussian 15), glare (saturated patch on eye area), occlude
(black band), out (face shifted out of frame), lowfps (drop frames to 8 fps).
Then run the pipeline and score: share of degraded frames held (and with the expected code),
share of clean frames wrongly held. Writes results/degrade.json.
"""
import argparse, json
from pathlib import Path
import cv2 as cv
import numpy as np

EXPECT = {"dark": "LOW_LIGHT", "blur": "MOTION", "glare": "GLARE", "occlude": "NO_EYE", "out": "NO_FACE", "lowfps": "LOW_FPS"}


def degrade(frame, kind):
    if kind == "dark":
        return (frame.astype(np.float32) * 0.15).astype(np.uint8)
    if kind == "blur":
        return cv.GaussianBlur(frame, (15, 15), 0)
    h, w = frame.shape[:2]
    if kind == "glare":
        f = frame.copy(); cv.circle(f, (w // 2, h // 3), max(6, w // 25), (255, 255, 255), -1); return f
    if kind == "occlude":
        f = frame.copy(); f[h // 4: h // 2, :] = 0; return f
    if kind == "out":
        return np.zeros_like(frame)  # face out of frame
    return frame


def write_variant(src, dst, kind, start, end):
    cap = cv.VideoCapture(src)
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out_fps = 8 if kind == "lowfps" else fps
    vw = cv.VideoWriter(dst, cv.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    i = 0; keep = max(1, round(fps / 8))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if kind == "lowfps":
            if i % keep == 0:
                vw.write(f)
        else:
            vw.write(degrade(f, kind) if start <= i < end else f)
        i += 1
    vw.release(); cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--start", type=int, default=90)
    ap.add_argument("--end", type=int, default=180)
    a = ap.parse_args()
    from holdsight.config import Thresholds
    from holdsight.pipeline import analyse
    t = Thresholds.load("config/thresholds.yaml")
    Path("results/degraded").mkdir(parents=True, exist_ok=True)
    report = {}
    for kind, code in EXPECT.items():
        dst = f"results/degraded/{kind}.mp4"
        write_variant(a.clip, dst, kind, a.start, a.end)
        r = analyse(dst, t)
        tl = r["timeline"]
        if kind == "lowfps":
            report[kind] = {"session": r["session"], "expected": code}
            continue
        deg = tl[a.start:a.end]; clean = tl[:a.start] + tl[a.end:]
        report[kind] = {
            "expected": code,
            "degraded_held_share": sum(x is not None for x in deg) / max(1, len(deg)),
            "degraded_right_code_share": sum(x == code for x in deg) / max(1, len(deg)),
            "clean_wrongly_held_share": sum(x is not None for x in clean) / max(1, len(clean)),
        }
    Path("results/degrade.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
