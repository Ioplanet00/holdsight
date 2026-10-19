# HoldSight — Technical Report (DRAFT)

**A value-or-hold eye-signal pipeline on OpenCV 5 and AWS Graviton (COOL)**
FunnyTech21 Co., Ltd. · Jiwoon Choi, Hyunsik Moon · OpenCV AI Competition 2026, powered by AWS
Status: draft. Every result is `[TO BE MEASURED]` until produced by the scripts in this repository.

> Screening signals only. Not a diagnosis, not a medical device, not vehicle control.

| Section | Judging criterion it answers |
|---|---|
| 1 Problem, users and contribution | Real-world impact (20) · Innovation (20) |
| 2 Architecture and data flow | Technical execution (30) · Documentation (10) |
| 3 OpenCV 5 implementation | Technical execution (30) |
| 4 AWS deployment, COOL and observability | Cloud delivery, reproducibility, responsible operation (10) · Best Use of COOL |
| 5 Evaluation | Technical execution (30) · Innovation (20) · Best Use of COOL |
| 6 User experience | User experience (10) |
| 7 Failure cases and limitations | Technical execution (30) |
| 8 Responsible use | Cloud delivery … responsible operation (10) |
| 9 Vision and roadmap | Real-world impact (20) · Innovation (20) |
| 10 Market and commercialisation | Real-world impact (20) |
| 11 Reproduce | Documentation (10) · Cloud delivery, reproducibility (10) |

## 1. Problem, users and contribution
Camera-based eye signals (blinks, eyelid closure) are used for driver-fatigue monitoring. The failure that hurts trust is not noise but **false confidence**: a number reported from a frame that could not be measured — face out of frame, glare on glasses, darkness, motion blur, collapsed frame rate. In a truck cab that means false alarms; in a screening setting it means false reassurance.

HoldSight is built around one rule: **every frame and every session is either a value or a hold with a reason code.** Contributions:
1. An all-OpenCV 5 driver eye-state pipeline (no proprietary model) with a two-level quality gate and seven reason codes.
2. **Hold correctness** as an evaluation target: we degrade clean clips in known places and measure whether the system refuses to answer in the right places, with the right reason.
3. A reproducible COOL-vs-pip comparison on the same Graviton4 instance type.

Users: fleet safety teams and driver-monitoring developers. Out of scope: diagnosis, identity, vehicle control.

## 2. Architecture and data flow
See `docs/architecture.png`. Browser → Amazon CloudFront (HTTPS) → FastAPI on EC2 Graviton4 (m8g) launched from the COOL AMI → OpenCV 5 pipeline → JSON + result page. Uploaded bytes are written only to a temporary file in tmpfs for the request and deleted in a `finally` block. CloudWatch receives metrics and logs without media. Evaluation scripts write JSON (no video) to `results/` and, on AWS, to a private S3 bucket.

## 3. OpenCV 5 implementation
3.1 Decode — `cv.VideoCapture`; timestamps from `CAP_PROP_POS_MSEC` give the delivered fps (`holdsight/decode.py`).
3.2 Face and eyes — `cv.FaceDetectorYN` with YuNet (OpenCV Zoo, MIT); face box, score, two eye-centre landmarks; eye box sized from inter-ocular distance (`face.py`).
3.3 Eye measures — crop → `cv.resize` → `cv.createCLAHE` → `cv.GaussianBlur` → Gaussian `cv.adaptiveThreshold` → `cv.morphologyEx` → `cv.findContours`; aperture = height/width of the dominant dark contour; pupil centre = `cv.moments`; glare = saturated-pixel ratio; blur = `cv.Laplacian` variance (`eye.py`).
3.4 Quality gate — frame: `NO_FACE`, `MOTION`, `LOW_LIGHT`, `GLARE`, `NO_EYE` (fixed precedence); session: `LOW_FPS`, `LOW_USABLE` (`gate.py`, `codes.py`). Thresholds in `config/thresholds.yaml`, set on our own consented recordings and frozen (git tag) before any Eyeblink8 run.
3.5 Driver profile — aperture normalised by the session open-eye baseline (90th percentile of usable frames); hysteresis (closed < 0.45, open ≥ 0.60), minimum 2 frames; events ≤ 0.5 s are blinks, longer are closures; blink rate per usable minute; PERCLOS over usable frames; longest closure. Held frames break events and are never counted as closed (`profiles/driver.py`).
3.6 Tests — unit tests for every hold code, blink/closure logic and held-frame handling; smoke test that a face-free clip is fully held (`tests/`).

## 4. AWS deployment, COOL and observability
Region `[TO BE CONFIRMED]`; instance m8g.4xlarge (COOL's recommended size) `[TO BE CONFIRMED]`; COOL version `[TO BE RECORDED]`; AMI ID `[TO BE RECORDED]`. CloudFormation `infra/holdsight.yaml`: EC2 (IMDSv2 required), CloudFront (HTTPS, caching disabled), IAM role limited to CloudWatch writes and `results/*` in one private, encrypted S3 bucket, security group open only to CloudFront. COOL evidence: `scripts/cool_evidence.py` saves `cv.getBuildInformation()`, version, AMI and instance type. Baseline: the same code with `opencv-python-headless==5.0.0.93` (pinned in `requirements.lock`) on the same instance type; an x86 instance for output comparison `[TO BE CONFIRMED]`.

## 5. Evaluation
5.1 Blink detection on Eyeblink8 (8 videos, 4 people, 70,992 frames, 408 annotated blinks; evaluation only). Frozen thresholds; interval-overlap matching; 95% CI by bootstrap over videos. Precision `[TO BE MEASURED]`, recall `[TO BE MEASURED]`, F1 `[TO BE MEASURED]`, frame hold-rate by code `[TO BE MEASURED]`.
5.2 Hold correctness (`eval/degrade.py`) on consented clips: darkening, blur, glare patch, occlusion, face out of frame, frame dropping to 8 fps. Degraded frames held `[TO BE MEASURED]`; held with the expected code `[TO BE MEASURED]`; clean frames wrongly held `[TO BE MEASURED]`.
5.3 COOL vs pip OpenCV 5 (`eval/bench.py`): same instance type, same inputs, ≥1,000 frames, 3 repeats. Median/p95 ms per frame and per operation (`resize`, `createCLAHE`, `adaptiveThreshold`, `findContours`, YuNet) `[TO BE MEASURED]`; throughput `[TO BE MEASURED]`; USD per 1,000 frames at the measured hourly price `[TO BE MEASURED]`.
5.4 Arm vs x86 (`eval/arm_vs_x86.py`): identical hold timeline `[TO BE MEASURED]`; max absolute difference of measures `[TO BE MEASURED]`.

## 6. User experience
One page, no install: consent check → upload → timeline (green = measured, colour = hold code) → measures or a session hold with a plain-English reason. Keyboard-operable controls; result region announced to screen readers (`aria-live`). Time to result `[TO BE MEASURED]`.

## 7. Failure cases and limitations
Failure gallery from consented recordings only (derived overlays): glasses glare, strong head turn, very low light, low frame rate, hand occlusion `[TO BE ADDED]`. Limitations: Eyeblink8 is small and RGB indoor; near-infrared input has not been evaluated; thresholds are hand-set and published; the aperture measure from contours is simpler than a learned landmark model.

## 8. Responsible use
Screening signals, not diagnosis; no medical-device or vehicle-control use. No identity recognition, no stored faces. Consent before own recordings and before upload. Videos deleted after processing. Public research data for evaluation only. EU driver-monitoring rules forbid biometric identification and require closed-loop processing; HoldSight does not identify people.

## 9. Vision and roadmap (goals, not achievements)
HoldSight is the open, honest-measurement layer of a larger goal: **one eye-signal engine for two screening questions** — mild cognitive impairment and driver impairment. Company background (not part of this submission): proprietary eye encoder 3.95 MB, median 1.82 ms on NVIDIA Jetson AGX Orin (eye encoder only, inference only); Korean patents KR 10-2987096 and KR 10-3010893 (granted 2026); an IRB-approved study at Jejung Hospital aiming to enrol about 160 adults aged 55+, **no results yet**. Goals: bring value-or-hold into in-vehicle and mobile products; evaluate near-infrared footage under licence; publish the hold-correctness benchmark; extend one screening layer across vehicles, healthcare and elder care.

## 10. Market and commercialisation
Company facts: FY2025 revenue KRW 320.5M; about KRW 150M in Jan–Oct 2026; purchase-linked R&D with Eisai Korea (KRW 570M, to March 2028); NVIDIA N-UP (KRW 148M); vehicle camera + radar validation with a global radar component maker; strategic investment from an automotive Tier-1 supplier; Brazil field demonstration and university MOUs; NeuronSense live on the App Store and Google Play; Gaze Strike in field pilot, launch planned. Market drivers with sources are listed in the README. Go-to-market goals: B2B driver monitoring (SDK licence + integration) → healthcare and research screening → elder care.

## 11. Reproduce
`make setup` → `make test` → freeze thresholds (`git tag thresholds-frozen`) → `make eval` → `make degrade` → `make deploy` → `make evidence` → `make bench` → `make destroy`.
