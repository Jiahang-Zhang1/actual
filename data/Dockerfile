FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    pyarrow \
    scikit-learn \
    requests \
    datasets \
    huggingface_hub

COPY ingest.py /app/
COPY data_generator.py /app/
COPY online_features.py /app/
COPY batch_pipeline.py /app/
COPY data_quality_check.py /app/
COPY feedback_collector.py /app/

CMD ["python3", "/app/data_quality_check.py"]
