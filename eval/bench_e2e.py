"""End-to-end timing of the full pipeline (decode + YuNet + eye stage + gate) on one clip.

  python -m eval.bench_e2e --label cool_prefetch --clip clip.mp4 --threads 4 --prefetch 16
"""
import argparse, json, os, time
from pathlib import Path
import cv2 as cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.threads:
        cv.setNumThreads(a.threads)
    os.environ["HOLDSIGHT_PREFETCH"] = str(a.prefetch)
    from holdsight.config import Thresholds
    from holdsight.face import FaceFinder
    from holdsight.pipeline import analyse
    t = Thresholds.load("config/thresholds.yaml")
    ff = FaceFinder(score_min=t.face_score_min)
    runs = []
    for _ in range(a.repeat):
        t0 = time.perf_counter(); r = analyse(a.clip, t, ff); wall = time.perf_counter() - t0
        runs.append({"frames": r["frames"], "wall_s": wall, "fps": r["frames"] / wall, "ms_per_frame": wall * 1000 / r["frames"]})
    res = {"label": a.label, "opencv": cv.__version__, "threads": cv.getNumThreads(), "prefetch": a.prefetch, "runs": runs}
    out = Path(a.out or f"results/bench_e2e_{a.label}.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2)); print(json.dumps(res))


if __name__ == "__main__":
    main()
