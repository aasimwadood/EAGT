# ============================
# EAGT — Dockerfile (multi-stage)
# ============================
# Base image: slim but with build tools for common ML deps
FROM python:3.10-slim AS base

# Prevents Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# System deps (ffmpeg/libsndfile for audio, git/curl for installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy source
COPY . /app

# Default ports: FastAPI 8000, Streamlit 8501
EXPOSE 8000 8501

# Healthcheck (FastAPI root)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/ || exit 1

# Entrypoint is configurable; by default, run the FastAPI server
# Override with `docker run ... <your command>` as needed.
CMD ["python", "scripts/serve.py", "--config", "configs/default.yaml"]
