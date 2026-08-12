# Engineering Rules

## Contract first

- `contracts/video_blueprint.schema.json` is the protocol source.
- `contracts/openapi.yaml` is the API contract.
- Do not silently alter schema semantics.

## Data integrity

- Dense arrays never go into the manifest.
- Missing values are explicit.
- Sidecars eventually require hashes, shape, axes, unit, coordinate space, and frame coverage.
- Writes should be atomic before a stage is marked succeeded.

## Modularity

Real analysis modules should eventually implement a common adapter contract and return artifacts, metrics, warnings, errors, and provenance.

## Reproducibility

Persist:
- code version
- adapter version
- weight hash
- config hash
- source hash
- upstream artifact hashes
- random seed
- device

## Offline / security

- production path is offline-first
- sanitize and resolve bundle-relative paths
- isolate untrusted media decode
- only approved model weights
- no user-supplied arbitrary pickle model loading

## Privacy

- anonymous same-video tracks only
- no face identity embeddings
- no real identity inference
- no age, health, emotion, or sexual-attribute inference

## Development discipline

- one Epic at a time
- every feature gets tests
- algorithm changes later require golden-video metrics/overlays
- partial failures are explicit, never silent
