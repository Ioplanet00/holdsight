"""Compare two analyse() JSON outputs of the same inputs (Arm vs x86): hold-code equality and max |diff|."""
import json, sys


def main(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    same_tl = a["timeline"] == b["timeline"]
    va, vb = a["session"].get("value", {}), b["session"].get("value", {})
    diffs = {k: abs((va.get(k) or 0) - (vb.get(k) or 0)) for k in ("blink_count", "blink_rate_per_min", "perclos", "longest_closure_s")}
    print(json.dumps({"identical_hold_timeline": same_tl, "max_abs_diff": diffs}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
