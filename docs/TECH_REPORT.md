# HoldSight — Technical Report

**A value-or-hold eye-signal pipeline on OpenCV and AWS Graviton4 (COOL)**
FunnyTech21 Co., Ltd. · Jiwoon Choi, Hyunsik Moon · OpenCV AI Competition 2026, powered by AWS
Status: v2, numbers measured 2026-10-20 on the live demo instance by the scripts in this repository. Arm vs x86 is not run yet.

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

Context: at FunnyTech21, one eye-signal engine does both cognitive-decline screening and driver-impairment detection. HoldSight is a separate, open demo build of that engine's measurement-honesty layer, and it contains none of the engine's code or weights.

## 2. Architecture and data flow
See `docs/architecture.svg`. Browser → Amazon CloudFront (HTTPS) → FastAPI on EC2 Graviton4 (m8g) launched from the COOL AMI → OpenCV pipeline (COOL build, OpenCV 4.14; the same code runs on pip OpenCV 5) → JSON + result page. Uploaded bytes are written only to a temporary file in tmpfs for the request and deleted in a `finally` block. CloudWatch receives metrics and logs without media. Evaluation scripts write JSON (no video) to `results/` and, on AWS, to a private S3 bucket.

## 3. OpenCV 5 implementation
3.1 Decode — `cv.VideoCapture` on a background thread with a 16-frame queue; timestamps from `CAP_PROP_POS_MSEC` give the delivered fps (`holdsight/decode.py`).
3.2 Face and eyes — `cv.FaceDetectorYN` with YuNet (OpenCV Zoo, MIT); face box, score, two eye-centre landmarks; eye box sized from inter-ocular distance. Frames wider than 720 px are downscaled for detection only, and landmarks are mapped back (`face.py`).
3.3 Eye measures — crop → `cv.resize` → `cv.createCLAHE` → `cv.GaussianBlur` → Gaussian `cv.adaptiveThreshold` → `cv.morphologyEx` → `cv.findContours`; aperture = height/width of the dominant dark contour; pupil centre = `cv.moments`; glare = saturated-pixel ratio and largest saturated blob (`cv.connectedComponentsWithStats`); covered eye = collapsed intensity spread and share of flat `cv.Sobel` gradient; blur = `cv.Laplacian` variance; blink signal = 5th-percentile grey level and dark-pixel share of the central eye box (`eye.py`).
3.4 Quality gate — frame: `NO_FACE`, `MOTION`, `LOW_LIGHT`, `GLARE`, `NO_EYE` (fixed precedence); session: `LOW_FPS`, `LOW_USABLE` (`gate.py`, `codes.py`). Darkness is judged on the face box (or the whole frame when no face is found), so a dark eye box on a well-lit face is reported as a covered eye, not `LOW_LIGHT`. Gate thresholds are set on our own consented recordings. `config/thresholds.yaml` is frozen by sha256 (`config/thresholds.frozen`).
3.5 Driver profile — two iris-darkness signals, each minus its 2 s rolling median and divided by a robust spread (1.4826 x MAD over 6 s), averaged into a z-score; a run with z > 3.5 is an event, with a 0.33 s refractory period; events ≤ 0.5 s are blinks, up to 3 s are closures. Parameters were chosen on Eyeblink8 dev videos 1-4, then frozen. The earlier lid-contour aperture detector over-counted during speech (dev F1 0.139) and was replaced; blink rate per usable minute; PERCLOS over usable frames; longest closure. Held frames break events and are never counted as closed (`profiles/driver.py`).
3.6 Tests — unit tests for every hold code, blink/closure logic and held-frame handling; smoke test that a face-free clip is fully held (`tests/`).

## 4. AWS deployment, COOL and observability
Region ap-northeast-2 (Seoul); instance m8g.xlarge (4 vCPU); COOL AMI "COOL-Graviton4-v2" (ami-052aa304c83bc70f3), which reports OpenCV 4.14.0-pre with the KleidiCV and carotene HALs, TBB and Arm Performance Libraries. CloudFormation `infra/holdsight.yaml`: EC2 (IMDSv2 required), CloudFront (HTTPS, caching disabled), IAM role limited to CloudWatch writes and `results/*` in one private, encrypted S3 bucket, security group open only to CloudFront, no SSH (management through SSM), encrypted root volume, systemd restart, EC2 auto-recover and reboot alarms, and a Route 53 health check on `/v1/health` with an SNS email alarm (`infra/health-alarm.yaml`, us-east-1). Uploads are limited to 30 MB video files and rate-limited per address. COOL evidence: `scripts/cool_evidence.py` saves `cv.getBuildInformation()`, version, AMI and instance type. Baseline: the same code with `opencv-python-headless==5.0.0.93` (pinned in `requirements.lock`) on the same instance type; an x86 instance for output comparison is planned.

## 5. Evaluation
5.1 Blink detection on Eyeblink8 (evaluation only, not redistributed). Interval-overlap matching; 95% CI by bootstrap over videos. Parameters chosen on dev videos 1-4; test videos 8-11 and talkingFace run once with the frozen file.

| Split | Precision | Recall | F1 (95% CI) |
|---|---|---|---|
| Dev 1-4, previous aperture detector | 0.154 | 0.126 | 0.139 (0.11-0.15) |
| Dev 1-4, iris-darkness detector | 0.751 | 0.842 | 0.794 (0.66-0.87) |
| Test 8-11 + talkingFace | 0.836 | 0.846 | 0.841 (0.70-0.90) |

Per test video F1: 8 = 0.82, 9 = 0.93, 10 = 0.90, 11 = 0.55 (recall 0.42), talkingFace = 0.87. Frame hold rate was at most 0.6% on test videos and 4.4% on dev video 3.

5.2 Hold correctness (`eval/degrade.py`) on a consented 60 s clip (960x540, 30 fps); frames 600-1200 degraded, all degradations simulated. Glare and occlusion are placed on the eye positions YuNet finds in the clean frame.

| Degradation | Expected | v1 right code | v2 right code | Clean frames wrongly held |
|---|---|---|---|---|
| Darkened x0.15 | `LOW_LIGHT` | 0% (held under another code) | 100% | 0 |
| Gaussian blur 15 | `MOTION` | 95.3% | 96.7% | 0 |
| Glare spot | `GLARE` | 0% | 99.8% | 0 |
| Band over eyes | `NO_EYE` | 0% | 100% | 0 |
| Face out of frame | `NO_FACE` | 100% | 100% | 0 |
| Noise σ 20 (new) | keep measuring | not run | 100% measured | 0 |
| 8 fps | `LOW_FPS` session | yes | yes | - |

5.3 COOL vs pip OpenCV 5 on the same m8g.xlarge (`eval/bench_matrix.py`, `eval/bench_e2e.py`).

Detection stage (YuNet + eye stage, 900 frames of the consented 960x540 clip), median ms per frame / aggregate fps:

| Threads x streams, scale | COOL | pip |
|---|---|---|
| 4 x 1, full | 12.09 / 82.6 | 15.26 / 65.5 |
| 4 x 1, half | 4.21 / 237 | 4.42 / 226 |
| 2 x 2, half | 6.45 / 310 | 7.13 / 280 |
| 1 x 4, half | 11.07 / 361 | 12.46 / 321 |

On 640x480 input the two builds are within about 5% at the detection stage, and pip leads in some configurations (2 x 2 half scale: 396 vs 451 fps).

End to end (decode + detection + eye stage + gate), Eyeblink8 video 1, 15,784 frames, 640x480:

| Config | COOL ms/frame (fps) | pip ms/frame (fps) |
|---|---|---|
| 4 threads, threaded decode | 10.15 (98.5) | 9.13 (109.6) |
| 4 threads, inline decode | 10.34 (96.7) | 9.29 (107.6) |
| 2 threads, threaded decode | 13.22 (75.7) | 13.00 (76.9) |
| 2 threads, inline decode | 13.63 (73.4) | 13.19 (75.8) |

Reading: COOL is faster only in the detection stage on the larger frame (about 21% at 4 threads). End to end, COOL was about 10% slower at 4 threads and about equal at 2. Our working explanation, which we have not proven, is that COOL is an OpenCV 4.14 build, while the pip wheel is OpenCV 5.0 with a newer DNN engine and decoder. For the most throughput per instance, run 1 thread per stream with 4 streams at half scale. Threaded decoding saves 2-3%. Live endpoint: a 60 s clip took 29.3 s round trip (server 27.8 s, 15.4 ms per frame). Cost at the on-demand price is about USD 0.22 per hour plus the COOL fee.

5.4 Arm vs x86 (`eval/arm_vs_x86.py`): not run yet.

## 6. User experience
One page, no install: consent check → upload → timeline (green = measured, colour = hold code) → measures or a session hold with a plain-English reason. Keyboard-operable controls; result region announced to screen readers (`aria-live`). Time to result for a 60 s clip: about 29 s through CloudFront.

## 7. Failure cases and limitations
- **Eyeblink8 video 11:** recall 0.42 (18 of 43 blinks found). The z-threshold chosen on the dev split appears too strict for this video; we have not tuned on it, because it is part of the test split.
- **Glare on glasses:** behind thick, reflective frames at low resolution, the iris contrast is too weak and blinks are under-counted. The reflection is diffuse rather than saturated, so `GLARE` does not fire either. This is a known gap: the gate catches saturated specular spots (99.8% in 5.2) but not low-contrast veiling glare.
- **False positives on video 8:** 13 extra events (precision 0.70); the cause has not been analysed yet.
- **Simulated degradations:** 5.2 uses simulated glare, occlusion, darkness and noise; real glasses glare and hand occlusion still need a consented failure gallery.
- **Limitations:** Eyeblink8 is small, RGB and indoor; near-infrared input has not been evaluated; thresholds are hand-set and published; the method is classical OpenCV, not a learned landmark model.

## 8. Responsible use
Screening signals, not diagnosis; no medical-device or vehicle-control use. No identity recognition, no stored faces. Consent before own recordings and before upload. Videos deleted after processing. Public research data for evaluation only. EU driver-monitoring rules forbid biometric identification and require closed-loop processing; HoldSight does not identify people.

## 9. Vision and roadmap (goals, not achievements)
HoldSight is the open, honest-measurement layer of a larger goal: **one eye-signal engine for two screening questions** — mild cognitive impairment and driver impairment. Company background (not part of this submission): proprietary eye encoder 3.95 MB, median 1.82 ms on NVIDIA Jetson AGX Orin (eye encoder only, inference only); Korean patents KR 10-2987096 and KR 10-3010893 (granted 2026); an IRB-approved study at Jejung Hospital aiming to enrol about 160 adults aged 55+, **no results yet**. Goals: bring value-or-hold into in-vehicle and mobile products; evaluate near-infrared footage under licence; publish the hold-correctness benchmark; extend one screening layer across vehicles, healthcare and elder care.

## 10. Market and commercialisation
Company facts: FY2025 revenue KRW 320.5M; about KRW 150M in Jan–Oct 2026; purchase-linked R&D with Eisai Korea (KRW 570M, to March 2028); NVIDIA N-UP (KRW 148M); vehicle camera + radar validation with a global radar component maker; strategic investment from an automotive Tier-1 supplier; Brazil field demonstration and university MOUs; NeuronSense live on the App Store and Google Play; Gaze Strike in field pilot, launch planned. Market drivers with sources are listed in the README. Go-to-market goals: B2B driver monitoring (SDK licence + integration) → healthcare and research screening → elder care.

## 11. Reproduce
`make setup` → `make test` → `make eval` (frozen thresholds) → `make degrade` → `make deploy ALARM_EMAIL=...` → `make health-alarm` → `make evidence` → `python -m eval.bench_matrix` / `python -m eval.bench_e2e` → `make destroy`.
