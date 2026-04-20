FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt /tmp/training-requirements.txt
COPY serving/requirements.txt /tmp/serving-requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /tmp/training-requirements.txt && \
    pip install requests pandas pyarrow

COPY data /workspace/data
COPY training /workspace/training
COPY scripts /workspace/scripts
COPY serving/tools /workspace/serving/tools
