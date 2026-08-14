You are the implementation and verification agent for Video Blueprint Studio M1.

Read `GEMINI.md` and `START_HERE.md` first, then inspect the repository before changing code.

Current phase: **Epic E11 — Integrated Production Acceptance**.

E0-E10 implementation already exists on `main`. Do not restart from E0, replace real stages with mocks, or disable implemented evidence stages merely to make an acceptance test pass.

Current objective: prove that one production job can run the implemented E2-E9 evidence stages together and produce an E10 physically verified Bundle from the same job.

Primary gate:
- `scripts/certify_full_stack_e11.py`
- `.github/workflows/e11-integrated-acceptance.yml`

Required integrated path:
real video -> normalization/shots/anonymous people -> pose/face/hands -> person masks + point tracks + dense flow -> camera motion -> body-local frame -> surface residual motion -> geometry-only micro-motion -> environment photometry -> schema/artifact validation -> verified bundle.zip

Do not claim success from schema validity alone. Require physical sidecars, checksums, stage success, privacy invariants, and post-write ZIP verification.

After the integrated gate is green, stop and report the evidence. The next phase is representative user-supplied real-video acceptance; do not invent qualitative accuracy claims without reviewing those outputs.
