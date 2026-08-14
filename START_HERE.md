# START HERE

Use Gemini / AI Studio as an implementation and verification agent for Video Blueprint Studio.

## Current phase

**Epic E12 — Representative Real-Video Acceptance (CURRENT).**

E0-E11 has reached `main`. E11 proved that the implemented E4-E9 evidence stages can run together in one production job and produce an E10 physically verified Bundle. Do not restart from E0 and do not interpret that integration result as proof of fine-detail extraction accuracy.

Read `GEMINI.md`, `docs/epic-plan.md`, the canonical contracts, and the current production pipeline before changing code.

## Current task

Run representative user-supplied real videos through the integrated production path, then evaluate both machine-readable evidence and actual frame-to-evidence alignment.

Use:

- `scripts/run_pipeline_job.py` to produce the real production Bundle;
- `scripts/evaluate_real_video_e12.py` to generate the evidence-first acceptance report;
- `packages/pipeline_core/real_video_acceptance.py` for diagnostic logic.

A stage marked `succeeded` is not sufficient acceptance. E12 must explicitly review low/zero module scores, disabled face/hand refinement, mask boundaries, tracking continuity, camera/body compensation, surface-motion evidence, micro-motion usability/rejection reasons, and artifact integrity.

Final acceptance remains `manual_frame_to_evidence_review_required` until the representative source frames and extracted evidence are reviewed together.
