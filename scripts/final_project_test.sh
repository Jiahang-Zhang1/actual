#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTUAL_URL="${ACTUAL_URL:-http://127.0.0.1:3001/accounts}"
SYNC_URL="${SYNC_URL:-http://127.0.0.1:5006/accounts}"
SERVING_URL="${SERVING_URL:-http://127.0.0.1:8000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://127.0.0.1:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://127.0.0.1:3000}"

wait_for_url() {
  local url="$1"
  local label="$2"
  local retries="${3:-60}"
  for _ in $(seq 1 "$retries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$label is ready: $url"
      return 0
    fi
    sleep 2
  done
  echo "$label did not become ready: $url" >&2
  return 1
}

open_url() {
  local url="$1"
  if command -v open >/dev/null 2>&1; then
    open -a "Google Chrome" "$url" >/dev/null 2>&1 || open "$url" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  else
    echo "Open manually: $url"
  fi
}

echo "Generating realistic bank import and serving batch data..."
python3 scripts/generate_actual_bank_import.py \
  --rows "${TEST_ROWS:-1500}" \
  --csv-output artifacts/test-data/actual_bank_transactions.csv \
  --qif-output artifacts/test-data/actual_bank_transactions.qif \
  --batch-output serving/runtime/batch_input_realistic.json \
  --training-output artifacts/test-data/synthetic_training_transactions.csv

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "Starting serving, Prometheus, and Grafana with Docker Compose..."
  (cd serving && python3 run.py monitor-up)
else
  echo "Docker Compose is not available; assuming serving/monitoring are already running."
fi

if ! curl -fsS "$SERVING_URL/readyz" >/dev/null 2>&1; then
  wait_for_url "$SERVING_URL/readyz" "serving"
fi

echo "Sending realistic predict_batch traffic to serving..."
curl -fsS \
  -H "Content-Type: application/json" \
  -d @serving/runtime/batch_input_realistic.json \
  "$SERVING_URL/predict_batch" >/tmp/actual_predict_batch_response.json

echo "Publishing data quality metrics into serving..."
python3 data/data_quality_check.py ingestion \
  --input artifacts/test-data/synthetic_training_transactions.csv \
  --output-json artifacts/test-data/ingestion_quality.json \
  --post-url "$SERVING_URL/monitor/data-quality" || true

echo "Exercising promotion and rollback scripts..."
python3 scripts/simulate_promotion_rollback.py

if ! curl -fsS http://127.0.0.1:3001 >/dev/null 2>&1; then
  echo "Starting Actual dev server in the background..."
  mkdir -p artifacts/test-data
  ACTUAL_ML_SERVICE_URL="$SERVING_URL" \
    NODE_ENV=development \
    BROWSER_OPEN=localhost:5006 \
    yarn start:server-dev > artifacts/test-data/actual-dev-server.log 2>&1 &
fi

wait_for_url http://127.0.0.1:3001 "Actual web" 90 || true
wait_for_url http://127.0.0.1:5006/health "Actual sync" 90 || true
wait_for_url "$PROMETHEUS_URL/-/ready" "Prometheus" 60 || true
wait_for_url "$GRAFANA_URL/api/health" "Grafana" 60 || true

echo "Opening project services in Chrome..."
open_url "$ACTUAL_URL"
open_url "$SYNC_URL"
open_url "$SERVING_URL/docs"
open_url "$SERVING_URL/monitor/summary"
open_url "$PROMETHEUS_URL"
open_url "$GRAFANA_URL"

cat <<EOF

Generated test data:
  artifacts/test-data/actual_bank_transactions.csv
  artifacts/test-data/actual_bank_transactions.qif

Use Actual's built-in flow:
  1. Open the manager screen.
  2. Click "Create test file".
  3. Open a checking account, then import artifacts/test-data/actual_bank_transactions.qif as bank transactions.
  4. Watch AI Suggestion and All Top-3 populate from serving.

Grafana login:
  admin / admin
EOF
