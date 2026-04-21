# Data Pipeline

This folder contains the data pipeline for the Smart Transaction Categorization system, deployed on Chameleon Cloud.

## Scripts

- `ingest.py` - Downloads and cleans the transaction dataset from HuggingFace, runs data quality check automatically after ingestion
- `data_generator.py` - Simulates live user traffic by sending synthetic transaction requests to the serving endpoint
- `online_features.py` - Computes real-time features for inference and monitors data drift
- `batch_pipeline.py` - Compiles versioned training and evaluation datasets from production data, runs quality check automatically
- `data_quality_check.py` - Data quality checks at three points: ingestion, training set compilation, and inference drift monitoring
- `feedback_collector.py` - Collects user feedback (predicted vs corrected category) and stores in PostgreSQL for retraining

## Dataset

[mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization) from HuggingFace, 4.5M transaction records, 12 categories.

## How to Run

```bash
python3 data_quality_check.py
python3 feedback_collector.py
python3 data_generator.py
python3 batch_pipeline.py
```

## Environment Variables

- `SERVING_URL` - Serving endpoint URL (default: http://129.114.26.190:30090/predict)
- `POSTGRES_HOST` - PostgreSQL host for feedback storage
- `POSTGRES_PORT` - PostgreSQL port (default: 5432)
- `POSTGRES_USER` - PostgreSQL username
- `POSTGRES_PASSWORD` - PostgreSQL password
- `POSTGRES_DB` - PostgreSQL database name
