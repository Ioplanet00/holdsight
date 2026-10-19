"""Eyeblink8 evaluation (EVALUATION ONLY - nothing is tuned here).

Precondition: config/thresholds.yaml is frozen (git tag `thresholds-frozen`) from the team's own
consented recordings. This script refuses to run if the working tree changed thresholds after the tag.
Eyeblink8 must be downloaded by the user from https://www.blinkingmatters.com/research; it is not
redistributed. Matching: a predicted blink matches a labelled blink if their frame intervals overlap.
95% CIs: bootstrap over videos (unit = video). Writes results/eyeblink8.json.

TODO(build): annotation parser for the dataset's .tag files (format to confirm on download).
"""
import argparse, json, random, subprocess
from pathlib import Path


def thresholds_frozen() -> bool:
    try:
        diff = subprocess.run(["git", "diff", "--quiet", "thresholds-frozen", "--", "config/thresholds.yaml"])
        return diff.returncode == 0
    except Exception:
        return False


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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    if not thresholds_frozen():
        raise SystemExit("thresholds are not frozen (tag thresholds-frozen) - refusing to evaluate")
    raise SystemExit("annotation parser not implemented yet - see TODO(build)")


if __name__ == "__main__":
    main()
