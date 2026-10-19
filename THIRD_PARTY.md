# Third-party components

| Component | Version | Licence | How it is used | Stored in this repo? |
|---|---|---|---|---|
| OpenCV (pip `opencv-python-headless`) | 5.0.0.93 | Apache-2.0 | baseline runtime | no (pip) |
| COOL — Cloud Optimized OpenCV for AWS Graviton4 | [TO BE RECORDED from the AMI] | AWS Marketplace terms (hourly software fee, 7-day trial) | Graviton4 runtime | no (AMI) |
| YuNet face detector `face_detection_yunet_2023mar.onnx` (OpenCV Zoo) | sha256 `8f2383e4…52fa4` | MIT (Copyright (c) 2020 Shiqi Yu) | face box + eye landmarks via `cv.FaceDetectorYN` | no (downloaded by `scripts/fetch_models.sh`) |
| FastAPI, Uvicorn, NumPy, PyYAML, pytest, python-multipart | see `requirements.lock` | MIT / BSD | service, tests | no (pip) |
| Eyeblink8 dataset (Fogelton & Benesova, blinkingmatters.com) | — | [TO BE CONFIRMED from the dataset's distribution terms] | evaluation only, cited, not redistributed | no |
