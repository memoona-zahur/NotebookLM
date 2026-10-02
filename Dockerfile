FROM python:3.12-slim

# Sentence-transformers pulls in torch, which needs libgomp for its CPU kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
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
COPY static ./static
COPY experiments ./experiments

# The app runs as a normal user. It needs the upload directory to be writable
# and nothing else, so the volume is mounted there rather than at /app.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/uploads "$HF_HOME" \
    && chown -R appuser:appuser /app /data "$HF_HOME"
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]