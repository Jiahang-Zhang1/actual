# Smart Transaction Categorization Team Testing Guide

This guide explains how to run the integrated system from the `serving`
branch, test the user flow inside Actual, inspect monitoring, and exercise the
promotion and rollback path.

## 1. Checkout

Run all commands from the repository root.

```bash
git fetch origin
git switch serving
git pull origin serving
yarn install
```

## 2. Start The Local Stack

```bash
bash scripts/final_project_test.sh
```

The script performs the local end-to-end setup:

- generates realistic bank transaction files
- starts serving, Prometheus, and Grafana when Docker Compose is available
- sends realistic `/predict_batch` traffic to serving
- publishes data quality metrics into serving
- exercises promotion and rollback scripts
- starts Actual dev server if it is not already running
- opens the main service pages in Chrome

Generated files:

```text
artifacts/test-data/actual_bank_transactions.qif
artifacts/test-data/actual_bank_transactions.csv
artifacts/test-data/synthetic_training_transactions.csv
serving/runtime/batch_input_realistic.json
```

Use the QIF file for Actual transaction import. The CSV file is useful for
inspection and data/training tests, but QIF is the smoother path for the Actual
account import flow.

## 3. Service URLs

| Service | URL | Purpose |
| --- | --- | --- |
| Actual frontend | http://127.0.0.1:3001 | Main user interface |
| Actual sync/server-dev | http://127.0.0.1:5006 | Local Actual server |
| Serving API docs | http://127.0.0.1:8000/docs | API contract and manual requests |
| Serving monitor summary | http://127.0.0.1:8000/monitor/summary | Current model and system behavior |
| Serving rollout decision | http://127.0.0.1:8000/monitor/decision | Promotion/rollback recommendation |
| Prometheus | http://127.0.0.1:9090 | Metrics query UI |
| Grafana overview | http://127.0.0.1:3000/d/actual-ml-system-overview/actual-ml-system-overview | System dashboard, login `admin / admin` |
| Grafana home | http://127.0.0.1:3000 | Dashboard index |

## 4. Test The Actual User Flow

1. Open http://127.0.0.1:3001.
2. On the file manager page, choose `Create test file`.
3. Do not use `Import file` on the manager page. That path imports an entire
   budget file, not bank transactions.
4. Open or create a checking account.
5. In the account page toolbar, select `Import`.
6. Import:

```text
artifacts/test-data/actual_bank_transactions.qif
```

7. After import, open the account transaction table.
8. Confirm the table includes:

```text
Category
AI Suggestion
All Top-3
Payment
Deposit
```

Expected behavior:

- `AI Suggestion` shows the selected prediction and confidence score.
- `All Top-3` shows three predicted category candidates and scores.
- The highest-confidence option is highlighted by default.
- Clicking another Top-3 option updates the highlight and transaction category.
- The click is recorded as feedback for monitoring and future retraining.

## 5. Manual Serving Checks

Health check:

```bash
curl -s http://127.0.0.1:8000/readyz
```

Single prediction:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_description": "Starbucks Store 1458 New York NY",
    "country": "US",
    "currency": "USD",
    "amount": -6.45,
    "transaction_date": "2026-04-20",
    "notes": "coffee"
  }' | jq
```

Batch prediction:

```bash
curl -s -X POST http://127.0.0.1:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d @serving/runtime/batch_input_realistic.json | jq '.items[0:3]'
```

Monitoring summary:

```bash
curl -s http://127.0.0.1:8000/monitor/summary | jq
```

Rollout decision:

```bash
curl -s http://127.0.0.1:8000/monitor/decision | jq
```

## 6. Data Quality Checks

Run an ingestion quality check and publish it to serving:

```bash
python3 data/data_quality_check.py ingestion \
  --input artifacts/test-data/synthetic_training_transactions.csv \
  --output-json artifacts/test-data/ingestion_quality.json \
  --post-url http://127.0.0.1:8000/monitor/data-quality
```

Inspect the result:

```bash
curl -s http://127.0.0.1:8000/monitor/summary | jq '.data_quality'
```

Prometheus metrics to check:

```text
actual_data_quality_pass
actual_data_quality_issue_count
actual_data_quality_metric
```

## 7. Promotion And Rollback

Run the simulation:

```bash
python3 scripts/simulate_promotion_rollback.py
```

Inspect the simulation output:

```bash
cat artifacts/test-rollout/archive/last_decision.json | jq
```

Dry-run the serving trigger:

```bash
python3 serving/tools/execute_rollout_action.py \
  --monitor-url http://127.0.0.1:8000/monitor/decision
```

Execute the recommended action when appropriate:

```bash
python3 serving/tools/execute_rollout_action.py \
  --execute \
  --monitor-url http://127.0.0.1:8000/monitor/decision \
  --reload-url http://127.0.0.1:8000/admin/reload-model
```

## 8. Monitoring Checklist

In Prometheus, query:

```text
request_count
request_latency_seconds
prediction_confidence
predicted_class_total
feedback_total
feedback_match_total
feedback_top3_match_total
actual_data_quality_pass
actual_data_quality_issue_count
actual_data_quality_metric
```

In Grafana, check dashboards for:

- system overview and service health
- serving request volume
- latency and error rate
- prediction confidence
- predicted category distribution
- feedback acceptance
- data quality status

## 9. System Flow

```text
Bank transaction import in Actual
        |
Actual stores transactions in SQLite
        |
Actual frontend loads account transactions
        |
Actual core calls serving /predict_batch
        |
Serving returns Top-3 category predictions
        |
Frontend shows AI Suggestion and All Top-3
        |
User selects or corrects a category
        |
Actual updates the transaction category
        |
Actual core records local feedback
        |
Serving receives /feedback for monitoring
        |
Data jobs check ingestion, training-set, and online drift quality
        |
Training builds a new dataset and evaluates a challenger model
        |
Promotion rules compare challenger and current deployed model
        |
Serving reloads promoted model or rolls back when metrics degrade
```

## 10. Component Ownership

Frontend / Actual:

```text
packages/desktop-client/src/components/accounts/Account.tsx
packages/desktop-client/src/components/transactions/TransactionsTable.tsx
packages/desktop-client/src/components/Titlebar.tsx
```

Actual core integration:

```text
packages/loot-core/src/server/ml/app.ts
packages/loot-core/src/server/ml/service.ts
packages/loot-core/src/server/ml/store.ts
packages/loot-core/src/server/transactions/app.ts
```

Serving:

```text
serving/app/main.py
serving/app/runtime.py
serving/tools/execute_rollout_action.py
serving/monitoring/prometheus-alerts.yml
serving/monitoring/grafana/dashboards/system_overview.json
serving/monitoring/grafana/dashboards/data_quality_monitoring.json
```

Data:

```text
data/data_quality_check.py
data/ingest.py
data/batch_pipeline.py
scripts/generate_actual_bank_import.py
```

Training:

```text
training/build_training_set.py
training/train_model.py
scripts/run_mlops_pipeline.py
scripts/promote_model.py
scripts/rollback_model.py
```

Kubernetes and automation:

```text
k8s/ml-system/base/
k8s/ml-system/overlays/staging/
k8s/ml-system/overlays/canary/
k8s/ml-system/overlays/production/
.github/workflows/mlops-automation.yml
docker/mlops.Dockerfile
```

## 11. Kubernetes Preview

Render manifests locally:

```bash
kubectl kustomize k8s/ml-system/overlays/staging
kubectl kustomize k8s/ml-system/overlays/canary
kubectl kustomize k8s/ml-system/overlays/production
```

Deploy on Chameleon:

```bash
kubectl apply -k k8s/ml-system/overlays/staging
kubectl apply -k k8s/ml-system/overlays/canary
kubectl apply -k k8s/ml-system/overlays/production
```

Port-forward production services:

```bash
kubectl -n actual-ml-production port-forward svc/actual-sync 5006:5006
kubectl -n actual-ml-production port-forward svc/smartcat-serving 8000:8000
kubectl -n actual-ml-production port-forward svc/prometheus 9090:9090
kubectl -n actual-ml-production port-forward svc/grafana 3000:3000
```

## 12. Common Mistakes

- `Import file` on the manager page imports an entire budget file.
- Bank transactions must be imported from inside an account page.
- Use the generated QIF file for the account transaction import path.
- Port `3001` is the Actual frontend.
- Port `5006` is the Actual local server.
- Port `8000` is serving.
- Port `9090` is Prometheus.
- Port `3000` is Grafana.
- Seeing Top-3 predictions in the Actual table confirms frontend, Actual core,
  and serving are connected.
- Seeing feedback/data-quality metrics in Prometheus confirms monitoring is
  connected.

## 13. Success Criteria

The local flow is working when:

- Actual opens on port `3001`.
- Serving is healthy on port `8000`.
- QIF transactions import into an account.
- `AI Suggestion` and `All Top-3` populate for imported transactions.
- Clicking a candidate updates the category.
- `/monitor/summary` reports request and feedback activity.
- Prometheus can query serving and data-quality metrics.
- Grafana dashboards load.
- Promotion/rollback simulation completes.
