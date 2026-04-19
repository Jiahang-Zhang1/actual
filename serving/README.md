# ActualBudget Serving

This repository serves the ActualBudget Smart Transaction Categorization model on Chameleon using a shared FastAPI API for three runtime variants:
- baseline sklearn
- ONNX
- ONNX dynamic quantization

## Zero-error bootstrap on a fresh Chameleon instance

If Docker is not installed yet:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
```

Then clone and bootstrap:

```bash
git clone <your-repo>
cd ActualBudget-Serving
bash bootstrap_chameleon.sh
```

The bootstrap script:
- creates `.env.example` if missing
- checks Docker and compose
- builds the image
- exports ONNX and dynamic-quantized ONNX artifacts
- starts the recommended serving option
- runs a smoke test

## Recommended serving option

```bash
python3 run.py up --variant onnx_dynamic_quant --workers 2
python3 run.py smoke
```

## Manual workflow

```bash
python3 run.py doctor
python3 run.py build
python3 run.py prepare
python3 run.py up --variant onnx_dynamic_quant --workers 2
python3 run.py smoke
```

## Main files

- `app/`: FastAPI service and backends
- `tools/prepare_artifacts.py`: source joblib -> ONNX -> dynamic quant artifacts
- `tools/benchmark_http.py`: online benchmark for `/predict` and `/predict_batch`
- `tools/benchmark_arrivals.py`: constant vs poisson arrival benchmark
- `docker/Dockerfile`: serving image
- `docker-compose.yml`: serve + tooling containers
- `run.py`: orchestration entrypoint

## Important fixes already included

- built-in `.env.example` fallback in `run.py`
- Docker prerequisite check with clear install instructions
- pinned compatible ONNX export/runtime versions
- locale support for ONNX Runtime (`en_US.UTF-8`)
- ONNX export sanitization for text vectorizers
- correct `/predict_batch` payload wrapping in `benchmark_http.py`
