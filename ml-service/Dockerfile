FROM python:3.11-slim

WORKDIR /app

COPY ml-service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ml-service /app/ml-service

ENV MODEL_DIR=/models/current

CMD ["uvicorn", "app:app", "--app-dir", "/app/ml-service", "--host", "0.0.0.0", "--port", "8000"]
