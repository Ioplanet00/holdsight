"""Controlled degradations for hold-correctness evaluation (template).

From a clean consented clip, write degraded copies whose degraded frame ranges are known:
dark (brightness x0.15), blur (Gaussian 15), glare (saturated specular spot on each detected eye),
occlude (dark band over both detected eyes), noise (Gaussian, sigma 20), out (camera aimed above the head, face leaves the frame),
lowfps (drop frames to 8 fps). All variants are simulated. Glare and occlusion are placed on the eye
positions YuNet finds in the clean frame, so they hit the eyes whatever the framing.
Then run the pipeline and score: share of degraded frames held (and with the expected code),
share of clean frames wrongly held. Writes results/degrade.json.
"""
import argparse, json
from pathlib import Path
import cv2 as cv
import numpy as np

EXPECT = {"dark": "LOW_LIGHT", "blur": "MOTION", "glare": "GLARE", "occlude": "NO_EYE", "out": "NO_FACE",
          "noise": None, "lowfps": "LOW_FPS"}   # noise: robustness check, values should keep coming

_finder = None


def _eyes(frame):
    global _finder
    if _finder is None:
        from holdsight.face import FaceFinder
        _finder = FaceFinder(score_min=0.5)
    f = _finder.find(frame)
    return None if f is None else (f.right_eye, f.left_eye, f.iod)


def degrade(frame, kind):
    if kind in ("glare", "occlude"):
        e = _eyes(frame)
        if e is None:
            return frame
        (rx, ry), (lx, ly), iod = e
        f = frame.copy()
        if kind == "glare":
            for x, y in ((rx, ry), (lx, ly)):
                cv.circle(f, (int(x + 0.05 * iod), int(y - 0.03 * iod)), max(3, int(0.09 * iod)), (255, 255, 255), -1)
        else:
            y0, y1 = int(min(ry, ly) - 0.25 * iod), int(max(ry, ly) + 0.25 * iod)
            x0, x1 = int(rx - 0.45 * iod), int(lx + 0.45 * iod)
            f[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = (35, 35, 35)
        return f
    if kind == "noise":
        n = np.random.default_rng(len(frame)).normal(0, 20, frame.shape)
        return np.clip(frame.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if kind == "dark":
        return (frame.astype(np.float32) * 0.15).astype(np.uint8)
    if kind == "blur":
        return cv.GaussianBlur(frame, (15, 15), 0)
    h, w = frame.shape[:2]
    if kind == "out":   # camera aimed above the head: top quarter of the frame stretched to full size
        return cv.resize(frame[: h // 4], (w, h), interpolation=cv.INTER_LINEAR)
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
        codes = {}
        for x in deg:
            codes[x or "measured"] = codes.get(x or "measured", 0) + 1
        report[kind] = {"degraded_codes": codes,
            "expected": code,
            "degraded_held_share": sum(x is not None for x in deg) / max(1, len(deg)),
            "degraded_right_code_share": sum(x == code for x in deg) / max(1, len(deg)),
            "clean_wrongly_held_share": sum(x is not None for x in clean) / max(1, len(clean)),
        }
    Path("results/degrade.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
