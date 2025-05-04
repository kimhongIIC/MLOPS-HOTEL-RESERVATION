# FROM python:3.11-slim 

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# WORKDIR /app

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgomp1 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# COPY . . 

# RUN pip install --no-cache-dir -e .

# RUN python pipeline/training_pipeline.py 

# EXPOSE 8080

# CMD ["python", "application.py"]

# ─── STAGE 1: Builder ───────────────────────────────────────
FROM python:3.11-slim AS builder

# avoid .pyc files, unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# install build-time deps (for compiling or training)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       build-essential libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire repo
COPY . .

# Run your training pipeline (drops artifacts into artifacts/model/)
RUN python pipeline/training_pipeline.py


# ─── STAGE 2: Runtime ───────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOKY_MAX_CPU_COUNT=4

WORKDIR /app

# Only runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy just what you need to serve:
#  - your Flask app
#  - config (with __init__.py)
#  - templates & static assets
#  - any helper code under src/
COPY application.py config/ templates/ static/ src/ ./

# Copy the trained model out of the builder stage
COPY --from=builder /app/artifacts/model /app/artifacts/model

# Run as non-root for security
RUN useradd --create-home appuser \
  && chown -R appuser /app
USER appuser

EXPOSE 8080

# Production WSGI server
CMD ["gunicorn", "application:app", "-b", "0.0.0.0:8080", "-w", "4"]
