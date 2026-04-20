FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas pyarrow requests

COPY data /app/data
COPY training/build_training_set.py /app/training/build_training_set.py

CMD ["python", "/app/data/data_generator.py", "--help"]
