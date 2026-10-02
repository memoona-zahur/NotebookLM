# Build the React frontend first, in a stage that never reaches the runtime
# image: node is a build-time tool only, and shipping it would add ~200 MB.
FROM node:22-alpine AS build

WORKDIR /build
# Copied separately so `npm ci` is cached until the dependencies themselves
# change, not on every source edit. The source is copied to the same root so
# `npm run build` runs with frontend/ as its project directory, exactly as it
# does on a developer machine.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# Runtime: Python and the built bundle, nothing else.
FROM python:3.12-slim

# Sentence-transformers pulls in torch, which needs libgomp for its CPU kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# Dependencies first: this layer is reused whenever only application code
# changes, so edits do not trigger a multi-minute torch reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY experiments ./experiments
COPY alembic.ini ./
COPY migrations ./migrations
# The compiled frontend. Vite writes to /static because that is what FastAPI
# already serves, so the image ships the same assets a local build produces.
COPY --from=build /static ./static

# The app runs as a normal user. It needs the upload directory to be writable
# and nothing else, so the volume is mounted there rather than at /app.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/uploads "$HF_HOME" \
    && chown -R appuser:appuser /app /data "$HF_HOME"
USER appuser

EXPOSE 8000

# No healthcheck here on purpose: Compose owns it, and duplicating the probe in
# both places is how they drift apart.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]