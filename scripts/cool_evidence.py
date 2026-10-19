"""Save evidence of which OpenCV build runs the workload (COOL vs pip baseline).

Writes results/<label>_evidence.txt with: cv.__version__, cv.getBuildInformation(),
platform/CPU, instance metadata (if on EC2, via IMDSv2), and the COOL version string if provided.
Usage: python scripts/cool_evidence.py --label cool --cool-version "<as shown in AWS Marketplace>"
"""
import argparse
import datetime as dt
import platform
import urllib.request
from pathlib import Path
import cv2 as cv


def imds(path: str) -> str:
    try:
        tok_req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                         headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        tok = urllib.request.urlopen(tok_req, timeout=1).read().decode()
        req = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{path}",
                                     headers={"X-aws-ec2-metadata-token": tok})
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except Exception:
        return "n/a (not on EC2)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["cool", "pip", "x86"])
    ap.add_argument("--cool-version", default="n/a")
    a = ap.parse_args()
    out = Path("results") / f"{a.label}_evidence.txt"
    out.parent.mkdir(exist_ok=True)
    lines = [
        f"utc: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"label: {a.label}",
        f"cool_version: {a.cool_version}",
        f"cv2.__file__: {cv.__file__}",
        f"cv.__version__: {cv.__version__}",
        f"platform: {platform.platform()}  machine: {platform.machine()}",
        f"instance_type: {imds('instance-type')}",
        f"ami_id: {imds('ami-id')}",
        f"region: {imds('placement/region')}",
        "", "==== cv.getBuildInformation() ====", cv.getBuildInformation(),
    ]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
