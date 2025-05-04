FROM python:3.11-slim 

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . . 

RUN pip install --no-cache-dir -e .

RUN python pipeline/training_pipeline.py 

# Production WSGI server
CMD ["gunicorn", "application:app", "-b", "0.0.0.0:8080", "-w", "4"]
