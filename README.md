# Video Blueprint Studio

Video Blueprint Studio is an offline-first video-analysis and verifiable-blueprint system. It converts an authorized source video into a structured **Video Blueprint bundle** containing source timing, anonymous character evidence, pose/landmarks, masks, motion, camera/body-local transforms, surface residuals, geometry-only micro-motion, environment photometry, quality, provenance, and physically verified sidecars.

The project does **not** perform identity replacement or video generation, and it must not perform biometric identity recognition or sensitive-attribute inference.

## Current status

Implemented and certified through **E11 Integrated Production Acceptance**. Current work is **E12 — Representative Real-Video Acceptance**.

E11 proved that the implemented E4-E9 evidence stages can run together in one production job and produce an E10 physically verified Bundle. E12 now asks the more important question: **what details are actually captured accurately from representative real videos?**

See `START_HERE.md`, `GEMINI.md`, and `docs/epic-plan.md` for the current acceptance boundary.

## Production evidence path

```text
source video
  -> real media probe / CFR normalization / PTS mapping
  -> shots + anonymous people
  -> pose / configured face-hand evidence
  -> person masks + sparse point motion + dense flow
  -> 2D camera motion
  -> body-local frame
  -> residual surface motion
  -> geometry-only micro-motion
  -> environment photometry
  -> canonical schema validation
  -> recursive artifact integrity validation
  -> verified bundle.zip
```

Higher evidence stages are opt-in and fail closed through configuration gates. A disabled stage is not equivalent to an accepted detail extractor.

## Prerequisites

- Python 3.11+
- `ffmpeg` / `ffprobe`
- Node.js for the applet runtime
- Docker / Docker Compose when using the containerized path
- Approved/pinned CV model dependencies and weights for real higher-stage inference

## Local engineering checks

Install development dependencies:

```bash
python3 -m pip install -e .[dev]
```

Run the standard gates:

```bash
python3 scripts/validate_contracts.py
python3 -m pytest tests/ -v
ruff check .
mypy apps packages
```

Start the API locally:

```bash
uvicorn apps.api.main:app --reload --port 8000
```

API docs are then available at `/docs` on the local server.

## E11 integrated certification

The canonical integrated gate is:

```text
scripts/certify_full_stack_e11.py
.github/workflows/e11-integrated-acceptance.yml
```

It enables sparse point motion, dense flow, camera motion, body-local frame, surface residual motion, micro-motion, and environment photometry together, then requires canonical Blueprint validation, recursive artifact integrity, and post-write ZIP verification from the same job.

## E12 real-video acceptance

Generate a production Bundle with the normal production runtime, then create the machine evidence report:

```bash
python scripts/evaluate_real_video_e12.py \
  --bundle /path/to/bundle.zip \
  --output /path/to/e12_real_video_acceptance.json
```

The report distinguishes:

- `machine_gate=failed`: missing/failed core stages or broken physical evidence;
- `machine_gate=needs_review`: structurally valid output with quality risks such as low/zero module scores, disabled face/hands, or no usable micro-motion;
- `machine_gate=passed`: no machine diagnostics were triggered.

Even `machine_gate=passed` is **not final perceptual acceptance**. Final status remains `manual_frame_to_evidence_review_required` until source frames/video are compared with tracking, pose, mask, camera/body, surface/micro-motion and environment evidence.

## Validation model

Validation is layered:

- Bundle/security integrity
- canonical JSON Schema
- timing/reference/hash integrity
- module-specific evidence/quality checks
- fail-closed micro-motion quality gates
- integrated production acceptance
- representative real-video machine diagnostics
- manual frame-to-evidence review

A structurally valid Bundle is necessary but is **not** sufficient proof that a real video's fine details were extracted accurately.

## Canonical sources

1. `contracts/video_blueprint.schema.json`
2. `contracts/openapi.yaml`
3. `contracts/example_blueprint.json`
4. `docs/PRD_M1.md`
5. `docs/architecture.md`
6. `docs/epic-plan.md`
7. `docs/engineering-rules.md`
8. `docs/validation-rules.md`
