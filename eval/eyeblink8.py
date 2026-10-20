"""Eyeblink8 / talkingFace evaluation against the dataset's own blink annotations.

Protocol (no tuning on test data):
  * dev split   = Eyeblink8 videos 1, 2, 3, 4   (blink parameters may be chosen here)
  * test split  = Eyeblink8 videos 8, 9, 10, 11 and talkingFace (evaluated once, frozen thresholds)
  * config/thresholds.yaml must match the sha256 stored in config/thresholds.frozen before --split test
Matching: a predicted blink matches a labelled blink if their frame intervals overlap (one-to-one).
95% CIs: bootstrap over videos. Data is downloaded by the user from
https://www.blinkingmatters.com/research and is never stored in this repository.

  python -m eval.eyeblink8 --root /data/eyeblink8 --talking /data/talkingFace --split dev --cache results/eb8_cache
"""
import argparse, hashlib, json, random
from pathlib import Path

DEV = ["1", "2", "3", "4"]
TEST = ["8", "9", "10", "11"]


def thresholds_frozen(path="config/thresholds.yaml", stamp="config/thresholds.frozen") -> bool:
    p, s = Path(path), Path(stamp)
    if not s.exists():
        return False
    return hashlib.sha256(p.read_bytes()).hexdigest() == s.read_text().split()[0]


def read_tag(path):
    """Return labelled blinks as (start_frame, end_frame) from an annotation .tag file.
    Each frame line is frame_id:blink_id:...; blink_id -1 means no blink."""
    blinks, cur, start, last = [], None, None, None
    started = False
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if line == "#start":
            started = True
            continue
        if not started or not line or line.startswith("#"):
            continue
        parts = line.split(":")
        try:
            fid, bid = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            continue
        if bid != cur:
            if cur is not None and cur >= 0:
                blinks.append((start, last))
            cur, start = bid, fid
        last = fid
    if cur is not None and cur >= 0:
        blinks.append((start, last))
    return blinks


def match(pred, gold):
    """pred/gold: lists of (start_frame, end_frame). Greedy one-to-one overlap matching."""
    used, tp = set(), 0
    for ps, pe in pred:
        for j, (gs, ge) in enumerate(gold):
            if j not in used and ps <= ge and gs <= pe:
                used.add(j); tp += 1; break
    return tp, len(pred) - tp, len(gold) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bootstrap(per_video, n=2000, seed=0):
    rnd = random.Random(seed); out = []
    for _ in range(n):
        s = [rnd.choice(per_video) for _ in per_video]
        out.append(prf(sum(x[0] for x in s), sum(x[1] for x in s), sum(x[2] for x in s)))
    ci = lambda k: (sorted(o[k] for o in out)[int(0.025 * n)], sorted(o[k] for o in out)[int(0.975 * n)])
    return {"precision": ci(0), "recall": ci(1), "f1": ci(2)}


def videos(root, talking, split):
    root = Path(root)
    ids = DEV if split == "dev" else TEST
    out = []
    for i in ids:
        d = next(root.rglob(i)) if not (root / i).exists() else root / i
        avi = sorted(d.glob("*.avi"))[0]
        out.append((f"eb8_{i}", avi, avi.with_suffix(".tag")))
    if split == "test" and talking:
        tf = sorted(Path(talking).rglob("*.avi"))[0]
        out.append(("talkingFace", tf, tf.with_suffix(".tag")))
    return out


def series_for(name, avi, cache, t):
    c = Path(cache) / f"{name}.json" if cache else None
    if c and c.exists():
        return json.loads(c.read_text())
    from holdsight.pipeline import analyse
    r = analyse(str(avi), t, series=True)
    s = {"signal": r["signal"], "ts_ms": r["ts_ms"], "fps": r["delivered_fps"], "timeline": r["timeline"]}
    if c:
        c.parent.mkdir(parents=True, exist_ok=True); c.write_text(json.dumps(s))
    return s


def predicted_frames(s, t):
    from holdsight.profiles.driver import blink_events
    blinks, _, _ = blink_events(s["signal"], s["ts_ms"], t)
    ts = s["ts_ms"]
    idx = {v: i for i, v in enumerate(ts)}
    return [(idx[a], idx[b]) for a, b in blinks]


def _prefetch(args):
    import cv2 as cv
    cv.setNumThreads(1)
    name, avi, cache, t = args
    series_for(name, avi, cache, t)


def evaluate(vids, cache, t, jobs=1):
    if jobs > 1 and cache:
        from multiprocessing import Pool
        with Pool(jobs) as p:
            p.map(_prefetch, [(n, a, cache, t) for n, a, _ in vids])
    per, rows = [], {}
    for name, avi, tag in vids:
        s = series_for(name, avi, cache, t)
        gold = read_tag(tag)
        pred = predicted_frames(s, t)
        tp, fp, fn = match(pred, gold)
        held = sum(x is not None for x in s["timeline"]) / max(1, len(s["timeline"]))
        dur_min = (s["ts_ms"][-1] - s["ts_ms"][0]) / 60000.0 if len(s["ts_ms"]) > 1 else 0
        per.append((tp, fp, fn))
        p, r, f = prf(tp, fp, fn)
        rows[name] = {"gold": len(gold), "pred": len(pred), "tp": tp, "fp": fp, "fn": fn,
                      "precision": p, "recall": r, "f1": f, "held_share": held, "minutes": dur_min}
    tot = [sum(x[k] for x in per) for k in range(3)]
    p, r, f = prf(*tot)
    return {"videos": rows, "total": {"tp": tot[0], "fp": tot[1], "fn": tot[2], "precision": p, "recall": r, "f1": f},
            "ci95": bootstrap(per) if len(per) > 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--talking", default=None)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--cache", default="results/eb8_cache")
    ap.add_argument("--jobs", type=int, default=1, help="videos processed in parallel (needs --cache)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.split == "test" and not thresholds_frozen():
        raise SystemExit("thresholds are not frozen (config/thresholds.frozen) - refusing to evaluate the test split")
    from holdsight.config import Thresholds
    t = Thresholds.load("config/thresholds.yaml")
    res = evaluate(videos(a.root, a.talking, a.split), a.cache, t, a.jobs)
    res["split"] = a.split
    out = Path(a.out or f"results/eyeblink8_{a.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["total"], indent=2))


if __name__ == "__main__":
    main()
