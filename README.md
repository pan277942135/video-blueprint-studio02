# Video Blueprint Studio — Milestone 1 (Epic E1.1 Real Media Probe)

Video Blueprint Studio M1 is an offline-first video analysis and verifiable-blueprint system.
The system converts authorized source videos into structured, reproducible **Video Blueprint bundles** preserving timing, camera motion, anonymous character motion, pose, landmarks, masks, surface residual motion, micro-motion, quality, and provenance.

> **Status Notice**: The project is currently in **Epic E1.1: Real Media Probe**. Media metadata probing uses system `ffprobe`/`ffmpeg` CLI via subprocess. No heavy CV/AI models or identity replacement models are integrated in E1.1.

---

## 🚀 Epic E1 Runbook

### Prerequisites
- **Python**: 3.11+
- **System Dependencies**: `ffmpeg` and `ffprobe` binaries in PATH
- **Docker** & **Docker Compose** (optional for containerized setup)

### 1. Installation & Environment Setup
Clone the repository and install the project in editable mode with development dependencies:

```bash
python3 -m pip install -e .[dev]
```

### 2. Local API Server Execution
Start the FastAPI server locally:

```bash
uvicorn apps.api.main:app --reload --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`.

### 3. Running Verification & Quality Checks

#### Contract Validation Script
Validate positive and negative blueprint JSON fixtures against `contracts/video_blueprint.schema.json`:

```bash
python3 scripts/validate_contracts.py
```

#### Integration & Contract Tests
Run the test suite with pytest:

```bash
python3 -m pytest tests/ -v
```

#### Code Linting (Ruff)
Run Ruff linter:

```bash
ruff check .
```

#### Type Checking (Mypy)
Run Mypy static type checker across API and packages:

```bash
mypy apps packages
```

#### One-Step E0 Preflight Verification
Execute the master E0 preflight script which runs all contract, test, lint, and type checks:

```bash
bash scripts/e0_preflight.sh
```

---

## 🐳 Docker Setup

Build and run the API service via Docker Compose:

```bash
docker compose up --build
```

---

## 📜 Canonical Protocols

1. `contracts/video_blueprint.schema.json` — Canonical protocol source (Draft 2020-12 JSON Schema).
2. `contracts/openapi.yaml` — Canonical REST API specification.
3. `contracts/example_blueprint.json` — Canonical example fixture.
