You are the implementation and verification agent for Video Blueprint Studio M1.

Read `GEMINI.md` and `START_HERE.md` first, then inspect the repository before changing code.

Current phase: **Epic E12 — Representative Real-Video Acceptance**.

E0-E11 already exists on `main`. E11 proved integrated execution and physical Bundle integrity. Do not restart from E0, replace real stages with mocks, disable implemented evidence stages, or treat a green run as proof that fine video details were captured accurately.

Current objective: run representative user-supplied real videos through the production path and determine, from physical evidence plus frame-to-evidence review, what was actually captured and what was missed.

Primary tools:
- `scripts/run_pipeline_job.py`
- `scripts/evaluate_real_video_e12.py`
- `packages/pipeline_core/real_video_acceptance.py`

Required production path:
real video -> normalization/shots/anonymous people -> pose/face/hands where configured -> person masks + point tracks + dense flow -> camera motion -> body-local frame -> surface residual motion -> geometry-only micro-motion -> environment photometry -> schema/artifact validation -> verified bundle.zip

For each real-video run, preserve source SHA and Bundle evidence. Generate the E12 report and explicitly surface low/zero module scores, disabled face/hands, surface-region coverage, micro-motion usability/rejection reasons, artifact integrity and privacy invariants.

Do not claim success from schema validity, physical sidecars, or `succeeded` stage status alone. Final acceptance remains `manual_frame_to_evidence_review_required` until source frames/video are compared with extracted tracks, masks, pose, motion and environment evidence.
