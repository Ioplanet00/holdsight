#!/usr/bin/env bash
# Download YuNet (OpenCV Zoo, MIT licence) and verify its checksum. The model is not stored in git.
set -euo pipefail
mkdir -p models
URL=https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
OUT=models/face_detection_yunet_2023mar.onnx
SHA=8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
[ -f "$OUT" ] || curl -fsSL -o "$OUT" "$URL"
echo "$SHA  $OUT" | sha256sum -c -
