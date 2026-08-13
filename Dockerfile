FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by the API and E3.2 MediaPipe runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libegl1 \
    libgles2 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install the base application plus the explicitly approved E3.2 runtime.
# Model task files are never downloaded here; production supplies the pinned
# task artifacts locally and the runtime verifies their full SHA256 before use.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[e3_face_hands]"

# Copy repository source code
COPY contracts/ ./contracts/
COPY configs/ ./configs/
COPY packages/ ./packages/
COPY apps/ ./apps/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
