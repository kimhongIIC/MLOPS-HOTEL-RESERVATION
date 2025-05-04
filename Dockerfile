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

# ─── STAGE 1: Build & (Optional) Train ───────────────────────────────────────
FROM python:3.11-slim AS builder

# Don’t generate .pyc files, don’t buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install any build-time deps (e.g. if your training pipeline needs gcc)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       build-essential libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source (including your training pipeline)
COPY . .

# (Optional) Train your model here; output goes to /app/artifacts/models/
RUN python pipeline/training_pipeline.py


# ─── STAGE 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Cap Loky to avoid spurious warnings & runaway CPU use
    LOKY_MAX_CPU_COUNT=4

WORKDIR /app

# Install only runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY application.py config/ templates/ static/ src/ /app/

# Copy pre-trained model from builder (or, if you train offline, copy from local disk)
COPY --from=builder /app/artifacts/models/ /app/artifacts/models/

# Create and switch to a non-root user
RUN useradd --create-home appuser \
  && chown -R appuser /app
USER appuser

# Expose the same port Cloud Run (and your Jenkins tests) will hit
EXPOSE 8080

# Use Gunicorn for production—4 workers is a reasonable default
CMD ["gunicorn", "application:app", "-b", "0.0.0.0:8080", "-w", "4"]
