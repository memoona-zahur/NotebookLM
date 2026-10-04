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

# OCR language data, fetched here so a container can read a scan out of the box.
# Without it every scanned upload is refused with instructions to run the
# installer - correct, but a poor first run for something the image can fix.
# Build-time rather than first-upload-time because the app has no progress bar:
# a download halfway through indexing looks like a hang.
#
# The download failing fails the build. Shipping an image that quietly cannot
# read scans reintroduces exactly the silent failure this feature removed. Build
# with --build-arg OCR_LANGUAGES=none to skip it deliberately, which is also the
# escape hatch for building behind a firewall.
ARG OCR_LANGUAGES=eng
RUN if [ "${OCR_LANGUAGES}" = "none" ]; then \
        echo "skipping OCR language data (OCR_LANGUAGES=none)"; \
    else \
        python -m app.ocr --install ${OCR_LANGUAGES}; \
    fi \
    && mkdir -p /data/uploads "$HF_HOME" \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /data "$HF_HOME"
USER appuser

EXPOSE 8000

# No healthcheck here on purpose: Compose owns it, and duplicating the probe in
# both places is how they drift apart.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]