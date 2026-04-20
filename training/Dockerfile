FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pandas pyarrow

COPY training/build_training_set.py /app/training/build_training_set.py

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/app/training/build_training_set.py"]
