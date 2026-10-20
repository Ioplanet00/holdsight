"""Benchmark matrix for one OpenCV build: threads x detector input scale x concurrent streams.

Frames are decoded once into memory, so decode cost is excluded and every configuration sees the
same pixels. Per frame: YuNet on the (optionally downscaled) frame + the eye stage on both eyes.
Concurrent streams run as separate processes, each with its own detector, on the same frames;
the aggregate rate is total frames / wall time. Writes one JSON with every configuration measured.

  python -m eval.bench_matrix --label cool --clip clip.mp4 --frames 900 --threads 1 2 4 --scales 1.0 0.67 0.5 --streams 1 2 4
"""
import argparse, json, multiprocessing as mp, platform, time
from pathlib import Path
import cv2 as cv
import numpy as np


def load(clip, n):
    cap = cv.VideoCapture(clip); out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def run(frames, threads, scale, repeat):
    cv.setNumThreads(threads)
    from holdsight.config import Thresholds
    from holdsight.eye import crop, read_eye
    from holdsight.face import FaceFinder
    t = Thresholds.load("config/thresholds.yaml")
    ff = FaceFinder(score_min=t.face_score_min)
    det_ms, eye_ms, frame_ms, found = [], [], [], 0
    t_start = time.perf_counter()
    for _ in range(repeat):
        for bgr in frames:
            f0 = time.perf_counter()
            small = bgr if scale == 1.0 else cv.resize(bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            face = ff.find(small)
            f1 = time.perf_counter()
            if face is not None:
                found += 1
                k = 1.0 / scale
                gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
                side = t.eye_box_scale * face.iod * k * 2
                for e in (face.right_eye, face.left_eye):
                    c = crop(gray, (e[0] * k, e[1] * k), side)
                    if c is not None:
                        read_eye(c, t)
            f2 = time.perf_counter()
            det_ms.append((f1 - f0) * 1e3); eye_ms.append((f2 - f1) * 1e3); frame_ms.append((f2 - f0) * 1e3)
    wall = time.perf_counter() - t_start
    return {"frames": len(frame_ms), "wall_s": wall, "face_found_share": found / max(1, len(frame_ms)),
            "detect_ms_median": float(np.median(det_ms)), "detect_ms_p95": float(np.percentile(det_ms, 95)),
            "eye_ms_median": float(np.median(eye_ms)),
            "frame_ms_median": float(np.median(frame_ms)), "frame_ms_p95": float(np.percentile(frame_ms, 95))}


def dnn_forward(frames, threads, engine):
    """Forward pass of the YuNet network alone through cv.dnn, to separate DNN engine cost
    from the detector's pre/post-processing. engine is 'default', 'classic' or 'new' (OpenCV 5 only)."""
    cv.setNumThreads(threads)
    from holdsight.face import DEFAULT_MODEL
    if engine == "default":
        net = cv.dnn.readNet(str(DEFAULT_MODEL))
    else:
        net = cv.dnn.readNet(str(DEFAULT_MODEL), "", "", getattr(cv.dnn, "ENGINE_" + engine.upper()))
    h, w = frames[0].shape[:2]
    names = net.getUnconnectedOutLayersNames()
    ms = []
    for i, f in enumerate(frames[:300]):
        blob = cv.dnn.blobFromImage(f, 1.0, (w // 32 * 32, h // 32 * 32))
        t0 = time.perf_counter(); net.setInput(blob); net.forward(names); ms.append((time.perf_counter() - t0) * 1e3)
    return {"engine": engine, "threads": threads, "forward_ms_median": float(np.median(ms[10:])),
            "forward_ms_p95": float(np.percentile(ms[10:], 95))}


def _worker(args):
    clip, n, threads, scale, repeat = args
    return run(load(clip, n), threads, scale, repeat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--streams", type=int, nargs="+", default=[1])
    ap.add_argument("--engines", nargs="*", default=[], help="also time the raw YuNet forward pass: default classic new")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    frames = load(a.clip, a.frames)
    h, w = frames[0].shape[:2]
    res = {"label": a.label, "opencv": cv.__version__, "module": cv.__file__, "machine": platform.machine(),
           "cpus": mp.cpu_count(), "clip_size": [w, h], "frames_per_pass": len(frames), "repeat": a.repeat, "runs": []}
    run(frames[:60], 1, 1.0, 1)   # warm-up
    for th in a.threads:
        for sc in a.scales:
            r = run(frames, th, sc, a.repeat)
            r.update(threads=th, scale=sc, streams=1, aggregate_fps=r["frames"] / r["wall_s"])
            res["runs"].append(r); print(json.dumps({k: r[k] for k in ("threads", "scale", "streams", "frame_ms_median", "aggregate_fps")}), flush=True)
    res["dnn_forward"] = []
    for eng in a.engines:
        for th in a.threads:
            try:
                r = dnn_forward(frames, th, eng)
            except Exception as e:   # engine not available in this build
                r = {"engine": eng, "threads": th, "error": str(e).splitlines()[0][:200]}
            res["dnn_forward"].append(r); print(json.dumps(r), flush=True)
    for ns in a.streams:
        if ns == 1:
            continue
        for sc in a.scales:
            th = max(1, mp.cpu_count() // ns)
            ctx = mp.get_context("spawn")
            t0 = time.perf_counter()
            with ctx.Pool(ns) as pool:
                parts = pool.map(_worker, [(a.clip, a.frames, th, sc, a.repeat)] * ns)
            wall = time.perf_counter() - t0
            tot = sum(p["frames"] for p in parts)
            stream_wall = max(p["wall_s"] for p in parts)
            r = {"threads": th, "scale": sc, "streams": ns, "frames": tot, "wall_s": stream_wall,
                 "aggregate_fps": tot / stream_wall, "frame_ms_median": float(np.median([p["frame_ms_median"] for p in parts])),
                 "detect_ms_median": float(np.median([p["detect_ms_median"] for p in parts])),
                 "face_found_share": float(np.mean([p["face_found_share"] for p in parts])), "wall_incl_startup_s": wall}
            res["runs"].append(r); print(json.dumps({k: r[k] for k in ("threads", "scale", "streams", "frame_ms_median", "aggregate_fps")}), flush=True)
    out = Path(a.out or f"results/bench_matrix_{a.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
