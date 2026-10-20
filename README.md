# HoldSight

**Eye signals that say when they can't measure.**
One OpenCV pipeline on AWS Graviton4 (served by the COOL build, checked against pip OpenCV 5) that measures driver eye state — blinks, blink rate, PERCLOS, longest closure — and returns a reason code instead of a guess whenever it cannot measure.

> Screening signals for research and product development only. Not a diagnosis, not a medical device, not vehicle control.

Built by FunnyTech21 (Incheon, South Korea) for the OpenCV AI Competition 2026, powered by AWS. All code in this repository was written for the competition. Our company's proprietary engine, models and weights are **not** included.

**Core idea:** at FunnyTech21, one eye-signal engine does both cognitive-decline screening and driver-impairment detection. HoldSight is a separate, open demo build of the measurement-honesty layer of that engine. It does not contain the engine itself.

![Architecture](docs/architecture.png)

---

## Judging criteria → where to look

| Criterion (weight) | Where it is in this repo |
|---|---|
| Technical execution (30%) | `holdsight/` pipeline (decode → YuNet → eye ROI → CLAHE → adaptiveThreshold → contours/moments → two-level gate → driver profile), `tests/`, `eval/eyeblink8.py`, `eval/degrade.py`, [Technical report](docs/TECH_REPORT.md) §3, §5 |
| Innovation (20%) | Value-or-hold output contract with 7 reason codes; **hold correctness** is itself evaluated (`eval/degrade.py`) — report §1, §5.2 |
| Real-world impact (20%) | Driver fatigue/distraction and the EU driver-distraction warning mandate; vision and market sections below; report §1, §9, §10 |
| User experience (10%) | `web/index.html`: one page, consent check, timeline with held segments coloured by reason, plain-English explanations |
| Documentation & presentation (10%) | This README, [docs/TECH_REPORT.md](docs/TECH_REPORT.md), [docs/architecture.png](docs/architecture.png), demo video (link on Devpost) |
| Cloud delivery, reproducibility, responsible operation (10%) | `infra/holdsight.yaml` (CloudFormation), `requirements.lock`, `make` targets, CloudWatch and Route 53 health alarms with SNS email (`infra/health-alarm.yaml`), video deleted after processing, least-privilege IAM, `scripts/cool_evidence.py` |
| Best Use of COOL (special award) | Same pipeline on the same Graviton4 instance: COOL vs `opencv-python-headless==5.0.0.93` — `eval/bench_matrix.py` (threads x streams x scale), `eval/bench_e2e.py`, `scripts/cool_evidence.py`; results above |

## Quick start (local)

```bash
make setup     # venv + pinned requirements.lock (opencv-python-headless==5.0.0.93)
make test      # downloads YuNet (sha256-checked) and runs unit + smoke tests
make demo      # http://localhost:8000 — upload a 20–90 s face clip
```

## Output contract

Every frame and every session is **either** a value **or** a hold with a reason:

```json
{"session": {"value": {"blink_count": 0, "blink_rate_per_min": 0.0, "perclos": 0.0, "longest_closure_s": 0.0}},
 "timeline": [null, null, "GLARE", "GLARE", null]}
{"session": {"hold": "LOW_USABLE", "why": "Too few frames could be measured, so no session result is given."}}
```

| Code | Level | Meaning |
|---|---|---|
| `NO_FACE` | frame | no face, or detector score below threshold |
| `NO_EYE` | frame | face found, eye region not readable or covered (texture collapse on a well-lit face) |
| `GLARE` | frame | saturated-pixel ratio, or largest saturated blob, in the eye box above threshold |
| `LOW_LIGHT` | frame | face box (or whole frame, if no face is found) too dark |
| `MOTION` | frame | face moved too far between frames, or image blurred |
| `LOW_FPS` | session | delivered frame rate below minimum |
| `LOW_USABLE` | session | too few usable frames, or clip too short |

Held frames carry no value and are never interpolated. There is no combined score.

## Evaluate

```bash
# 1. Gate thresholds are set on our own consented recordings; blink parameters on Eyeblink8 dev videos 1-4.
#    config/thresholds.yaml is then frozen (sha256 in config/thresholds.frozen).
# 2. Download Eyeblink8 yourself (https://www.blinkingmatters.com/research) — evaluation only, not redistributed
make eval EYEBLINK8=/path/to/eyeblink8          # test split refuses to run if thresholds changed
make degrade CLIP=/path/to/consented_clip.mp4   # hold correctness under controlled degradations
```

Measured 2026-10-20 on the live instance (m8g.xlarge, 4 vCPU, Seoul). COOL = the OpenCV 4.14 build shipped in the COOL AMI; pip = `opencv-python-headless` 5.0.

**Blinks, Eyeblink8** (interval-overlap matching; 95% bootstrap CI over videos)

| Split | Precision | Recall | F1 |
|---|---|---|---|
| Dev, videos 1-4 (parameters chosen here) | 0.751 | 0.842 | 0.794 (CI 0.66-0.87) |
| Test, videos 8-11 + talkingFace (frozen, run once) | 0.836 | 0.846 | 0.841 (CI 0.70-0.90) |

The previous detector (lid-contour aperture) scored F1 0.139 on the same dev split and over-counted badly while people talked. The new one uses iris darkness with a robust z-score and a refractory period. On talkingFace (a person speaking) F1 is 0.872.

**Hold correctness** (consented clip, 600 degraded frames per variant, simulated)

| Degradation | Expected | Held with expected code | Clean frames wrongly held |
|---|---|---|---|
| Darkened x0.15 | `LOW_LIGHT` | 100% | 0 |
| Gaussian blur 15 | `MOTION` | 96.7% | 0 |
| Glare spot on each eye | `GLARE` | 99.8% | 0 |
| Dark band over both eyes | `NO_EYE` | 100% | 0 |
| Face out of frame | `NO_FACE` | 100% | 0 |
| Gaussian noise σ 20 | keep measuring | 100% measured, 0 held | 0 |
| 8 fps | `LOW_FPS` (session) | yes | - |

**COOL vs pip OpenCV 5** (same instance, same code, same inputs)

| Measurement | COOL | pip | Reading |
|---|---|---|---|
| Detection stage, 960x540, 4 threads, ms/frame | 12.09 | 15.26 | COOL ~21% faster |
| Detection stage, 960x540, 1 thread x 4 streams, half scale, aggregate fps | 361 | 321 | COOL ~12% more throughput |
| End to end (decode + detect + eye + gate), 640x480, 15,784 frames, 4 threads, ms/frame | 10.15 | 9.13 | **COOL ~10% slower** |
| End to end, 2 threads, ms/frame | 13.22 | 13.00 | about equal |

We claim a COOL advantage only for the detection stage. End to end, on our 640x480 test video, the pip OpenCV 5 wheel was about 10% faster. Decoding on a background thread saved 2-3%.

**Known failure cases**
- Eyeblink8 video 11: recall 0.42 (25 of 43 blinks missed).
- Glare on glasses: a thick-framed, reflective pair of glasses at low resolution leaves too little iris contrast. Blinks are then under-counted, and the `GLARE` code does not fire, because the reflection is diffuse rather than saturated.

Arm vs x86 output comparison: not run yet.

## Deploy to AWS

```bash
# Requires: AWS account, COOL (Cloud Optimized OpenCV for AWS Graviton4) Marketplace subscription
make deploy STACK=holdsight REGION=<region> COOL_AMI=<ami-id>
make evidence LABEL=cool COOL_VERSION="<version shown in Marketplace>"   # on the instance
make bench LABEL=cool CLIPS="data/bench/*.mp4" USD_PER_HOUR=<price>
make destroy STACK=holdsight REGION=<region>
```

---

## Why this matters — vision

HoldSight is a small, open slice of a bigger idea we are building at FunnyTech21: **one eye-signal engine for two screening questions** — early signs of cognitive decline (mild cognitive impairment) and driver impairment (fatigue, distraction). Both show up in the eyes; both need a measurement that is honest about when it cannot see.

What already exists (company background, not part of this submission):

- **Edge-light engine.** Our proprietary eye encoder is 3.95 MB and runs in a median 1.82 ms on NVIDIA Jetson AGX Orin (eye encoder only, inference only; capture and pre-processing excluded).
- **Granted patents (Korea):** KR 10-2987096 (granted 2026-07-01) — contactless autonomic-indicator analysis for early dementia screening; KR 10-3010893 (granted 2026-08-24) — multi-ROI biosignal measurement with personalised calibration.
- **Clinical validation in progress:** an IRB-approved study at Jejung Hospital aims to enrol about 160 adults aged 55 and over. **There are no results yet.** HoldSight uses none of its data.

Where we want to take it (goals, not achievements):

| Horizon | Goal |
|---|---|
| Next | Bring the value-or-hold contract from HoldSight into our in-vehicle and mobile products; evaluate near-infrared driving footage under proper licences |
| Then | Publish a hold-correctness benchmark so any eye-tracking pipeline can be tested on *when it should refuse to answer* |
| Vision | The same camera-based screening layer across **vehicles, healthcare and elder care** — a car cabin, a clinic tablet and a care-home screen reporting the same signals with the same honesty rules |

## Market and commercialisation

Demand we already see (company facts):

- **Revenue:** KRW 320.5M in FY2025 (financial statements); about KRW 150M in January–October 2026.
- **Purchase-linked R&D with Eisai Korea:** KRW 570M, running to March 2028.
- **NVIDIA N-UP:** KRW 148M (2026, in-vehicle driver monitoring).
- **Automotive:** in-vehicle camera + radar validation with a global radar component maker (March 2026 – March 2027), and a strategic investment from an automotive Tier-1 supplier.
- **Brazil:** field demonstration of our driver monitor at MinasParts 2026, and MOUs with Mackenzie Presbyterian University and IBGP.
- **Products:** NeuronSense — 80-second eye check, screening not diagnosis — on the [App Store](https://apps.apple.com/app/id6806928763) and [Google Play](https://play.google.com/store/apps/details?id=com.funnytech21.neuronsense). Gaze Strike, touch-free gaze games for older adults — field pilot in progress (Ganghwa, 23 October 2026), launch planned.

Why the market is moving (official sources):

- **Regulation:** in the EU, Advanced Driver Distraction Warning (ADDW) is required for all new vehicles from 7 July 2026, and the rules forbid biometric identification and require closed-loop processing ([Regulation (EU) 2019/2144](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02019R2144-20240707), [Delegated Regulation (EU) 2023/2590](https://eur-lex.europa.eu/eli/reg_del/2023/2590)). In the US, the impaired-driving prevention technology rule is still at the advance-notice stage ([Federal Register, 2024-01-05](https://www.federalregister.gov/documents/2024/01/05/2023-27665/advanced-impaired-driving-prevention-technology)).
- **Road safety:** 3,275 people were killed in distraction-affected crashes and 633 in drowsy-driving crashes in the US in 2023 ([NHTSA 813703](https://crashstats.nhtsa.dot.gov/Api/Public/Publication/813703), [NHTSA 813705](https://crashstats.nhtsa.dot.gov/Api/Public/Publication/813705)); fatigue is estimated to be involved in 15–20% of crashes in the EU ([ERSO 2023](https://road-safety.transport.ec.europa.eu/system/files/2024-01/ERSO_SPI_Fatigue_2023-12-13.pdf)).
- **Ageing:** 57 million people lived with dementia worldwide in 2021 ([WHO fact sheet](https://www.who.int/news-room/fact-sheets/detail/dementia)).
- **Driver-monitoring market size:** published estimates differ widely by scope, from USD 2.08B to USD 5.6B for 2030 ([TechSci](https://www.techsciresearch.com/report/driver-monitoring-systems-dms-market/3235.html), [MarkNtel](https://www.marknteladvisors.com/research-library/global-automotive-driver-monitoring-system-market.html)).

Go-to-market plan (goals):

1. **B2B driver monitoring first** — automotive suppliers and fleets, where regulation already creates demand. Revenue model: SDK licence per device or per vehicle programme, plus integration projects.
2. **Healthcare and research next** — screening tools for clinics and research partners, building on the ongoing IRB study (screening, not diagnosis).
3. **Elder care** — tablet-based, touch-free engagement and screening for care settings, starting from the Gaze Strike field pilot.

## Responsible use

- Screening signals only: not a diagnosis, not a medical device, not vehicle control.
- No identity recognition; no face images stored.
- Uploaded videos are processed in a temporary file in memory (`/dev/shm` when available) and deleted right after processing; only derived numbers are returned.
- Own test recordings come only from team members who signed consent (personal data, sensitive data, research and commercial use, publication).
- Eyeblink8 is used for evaluation only, cited and not redistributed. Blink parameters were chosen on dev videos 1-4 and then frozen; videos 8-11 and talkingFace are kept as a held-out test.

## Third-party components

See [THIRD_PARTY.md](THIRD_PARTY.md). YuNet (OpenCV Zoo, MIT) is downloaded at setup and checksum-verified, not stored in git.

## Licence

Apache License 2.0 (candidate — see `LICENSE`). Covers only the code in this repository.

## Team

- **Jiwoon Choi**, Co-founder & CEO, FunnyTech21 — engine development, model lightweighting and advancement of core markers (vision pipeline, evaluation).
- **Hyunsik Moon**, Engineer, FunnyTech21 — back-end, system integration and servers (AWS deployment, web app, reproducibility).
