FROM python:3.11-slim

WORKDIR /app

COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY training /app/training

CMD ["python", "/app/training/train_model.py", "--help"]
