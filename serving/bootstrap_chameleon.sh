#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found. Installing docker.io and docker compose plugin..."
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-v2
  sudo systemctl enable --now docker
  sudo usermod -aG docker "$USER" || true
  echo "Docker installed. Re-login or run: newgrp docker"
fi

mkdir -p results/raw results/summary models/optimized artifacts/examples

python3 run.py doctor
python3 run.py build
python3 run.py prepare
python3 run.py up --variant onnx_dynamic_quant --workers 2
python3 run.py smoke

echo "Bootstrap complete. Service is running."
