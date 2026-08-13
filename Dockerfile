FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by the API and optional E3.2 MediaPipe runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy repository source code
COPY contracts/ ./contracts/
COPY configs/ ./configs/
COPY packages/ ./packages/
COPY apps/ ./apps/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
