# Dockerfile for data pipeline
# Assisted by Claude Sonnet 4.6

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    pandas \
    pyarrow \
    scikit-learn \
    psycopg2-binary \
    requests \
    datasets \
    huggingface_hub

# Copy all scripts
COPY ingest.py /app/
COPY data_generator.py /app/
COPY online_features.py /app/
COPY batch_pipeline.py /app/
COPY data_quality_check.py /app/
COPY feedback_collector.py /app/

CMD ["python3", "/app/data_quality_check.py"]
