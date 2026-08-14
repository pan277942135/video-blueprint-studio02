# START HERE

Use Gemini / AI Studio as an implementation and verification agent for Video Blueprint Studio.

## Current phase

**Epic E11 — Integrated Production Acceptance (CURRENT).**

E0-E10 implementation has reached `main`. Do not restart the project from E0 and do not remove real CV stages that are already certified individually.

Read `GEMINI.md`, `docs/epic-plan.md`, the canonical contracts, and the current production pipeline before changing code.

## Current task

Prove one production job can run the implemented evidence stack together:

`real media -> E1 media -> E2 people -> E3 pose/face/hands -> E4 masks/point/dense motion -> E5 camera -> E6 body-local -> E7 surface motion -> E8 micro-motion -> E9 environment -> E10 verified bundle`

Use `scripts/certify_full_stack_e11.py` and `.github/workflows/e11-integrated-acceptance.yml` as the acceptance gate.

Do not claim production readiness merely because schema validation, unit tests, or isolated stage certifications pass. E11 requires integrated evidence and a physically verified bundle from the same job.

After E11 integrated certification passes, the next gate is representative user-supplied real-video acceptance and qualitative/quantitative review of extracted evidence.
