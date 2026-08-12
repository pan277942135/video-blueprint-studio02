# Epic Plan

## E0 — Contract and engineering skeleton [CURRENT]

Deliver:
- repo structure
- canonical schema wiring
- Pydantic/schema model layer
- OpenAPI FastAPI stubs
- deterministic mock pipeline
- job model
- bundle exporter
- contract tests
- Docker/CI skeleton
- runbook

Acceptance:
- upload -> create analysis -> mock complete -> blueprint -> validation -> ZIP
- example fixture passes schema
- intentionally broken fixtures fail
- API contract tests pass
- no real CV model dependencies

## E1 — Media and jobs

Add real ffprobe/FFmpeg media probe, source validation, normalization, PTS map, cache/retry/cancel/export.

## E2 — Shots and people

PySceneDetect + RTMDet + anonymous track association + bbox overlay.

## E3 — Pose / face / hands

RTMW + MediaPipe refinement + time-series sidecars + overlays.

## E4 — Mask / point / flow

SAM 2 + TAPIR/TAPNext + SEA-RAFT.

## E5 — Camera

Background tracks + RANSAC 2D camera estimation + stabilization + quality.

## E6 — Surface and micro-motion

Body-local frame + ROI + residual surface signals + leakage/coherence/quality gate.

## E7 — Review Studio

Synchronized player, layers, curves, revisions, downstream-only reruns.

## E8 — Hardening

Performance, security, offline installation, SBOM, model/license registry, acceptance matrix.
