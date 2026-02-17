# ---- Stage 1: Build ----
FROM python:3.13-slim AS builder

WORKDIR /build

# gcc needed to compile C extensions (e.g. soundfile, onnxruntime)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.13-slim

WORKDIR /app

# Runtime-only system libs: audio processing + Playwright Chromium deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libxshmfence1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder (no gcc carried over)
COPY --from=builder /install/lib /usr/local/lib
COPY --from=builder /install/bin /usr/local/bin

# Install Playwright Chromium (separate layer — rarely changes)
RUN playwright install chromium --with-deps && \
    rm -rf /var/lib/apt/lists/*

# Download turn-detector ONNX model (depends on packages, not app code)
# Only copy the files needed to run download-files so code changes don't
# invalidate this layer.
COPY agent_worker.py agent_lifecycle.py ./
RUN python agent_worker.py download-files && \
    rm agent_worker.py agent_lifecycle.py

# Copy app code (changes here don't re-download model or chromium)
COPY . .

RUN mkdir -p logs transcripts

EXPOSE 8080

CMD ["python", "app.py"]
