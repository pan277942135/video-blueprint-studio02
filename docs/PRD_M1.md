# PRD M1 — Condensed Engineering Baseline

## Product boundary

M1 turns an authorized source video into a structured, verifiable Video Blueprint. It does not replace the person and does not synthesize the final video.

## Required output family

- Blueprint Manifest (`blueprint.json`)
- high-density sidecars
- overlays
- quality report
- provenance
- license report
- export bundle ZIP

## Core functional areas

- ingest/probe/normalize
- shot detection
- anonymous person detection/tracking
- masks
- whole-body pose
- face and hands
- point tracks / dense flow
- camera motion
- body-local coordinate frame
- surface residual motion
- micro-motion
- environment proxies
- optional semantics
- validation
- debug rendering
- export

## Default source limits

- 2–60 seconds
- <= 2 GiB
- decodable MP4/MOV/MKV/WebM
- source fps roughly 15–60
- 1–4 primary visible people
- larger source videos may use proxy analysis while preserving coordinate mapping

## Data rule

The manifest stores logical metadata and references. Dense time-series data belongs in sidecars such as NPZ/Parquet/JSONL/RLE/PNG sequence.

## Privacy rule

M1 uses anonymous `character_id` tracks. It does not infer a real person's identity, age, health, emotion, or sexual attributes and does not export biometric face embeddings.

## Micro-motion definition

Micro-motion means low-amplitude visible surface or garment residual motion after camera and whole-body/segment motion compensation. It is a visual constraint signal, not a physiological diagnosis.
