#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

echo "== /predict single cases =="
for f in \
  single_starbucks.json \
  single_payroll.json \
  single_comcast.json \
  single_donation.json \
  single_bankfee.json \
  single_shopping.json \
  single_clean_fields.json
  do
  echo "\n--- $f ---"
  curl -sS -X POST "$BASE_URL/predict" \
    -H 'Content-Type: application/json' \
    --data-binary "@/mnt/data/$f"
  echo
 done

echo "\n== /predict_batch mixed 4 =="
curl -sS -X POST "$BASE_URL/predict_batch" \
  -H 'Content-Type: application/json' \
  --data-binary @/mnt/data/batch_mixed_4.json

echo "\n\n== /predict_batch mixed 8 =="
curl -sS -X POST "$BASE_URL/predict_batch" \
  -H 'Content-Type: application/json' \
  --data-binary @/mnt/data/batch_mixed_8.json

echo "\n\n== /feedback example =="
curl -sS -X POST "$BASE_URL/feedback" \
  -H 'Content-Type: application/json' \
  -d '{
    "transaction_id": "tx-demo-001",
    "model_version": "v2_tfidf_linearsvc",
    "predicted_category_id": "Shopping & Retail",
    "applied_category_id": "Shopping & Retail",
    "confidence": 0.324538,
    "candidate_category_ids": ["Shopping & Retail", "Financial Services", "Charity & Donations"]
  }'

echo "\n\n== /monitor/summary =="
curl -sS "$BASE_URL/monitor/summary"
