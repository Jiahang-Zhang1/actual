# Smart Transaction Categorization System Guide

This is the single handoff document for the integrated Actual Budget smart
transaction categorization system. It combines the serving service inventory,
team testing guide, deployment handoff notes, monitoring plan, and project
requirement traceability into one place.

Use this document when a teammate switches to the `serving` branch and wants to
understand, run, test, or explain the system from end to end.

## 1. What The System Does

The project adds transaction category suggestions directly into the regular
Actual Budget account transaction flow.

A user imports bank transactions, opens an account table, and sees:

- `Category`: the current Actual category, or the best available prediction when the row is still uncategorized
- `AI Suggestion`: the selected predicted category with its confidence score
- `All Top-3`: three candidate categories, each with a confidence score

The user can click any Top-3 candidate. That action updates the transaction
category and records feedback for monitoring and future retraining.

The system is designed to connect four parts:

```text
Actual frontend -> Actual core bridge -> serving inference API -> monitoring and MLOps automation
```

The longer operating loop is:

```text
Bank transaction import
        |
Actual stores transactions in SQLite
        |
Actual account page requests batch predictions
        |
Serving returns Top-3 predicted categories
        |
User reviews or corrects the suggestion
        |
Feedback is stored locally and mirrored to serving
        |
Monitoring tracks output, latency, errors, feedback, and data quality
        |
Training builds/evaluates a challenger model
        |
Promotion rules publish a better model
        |
Rollback rules restore the previous model if production degrades
```

## 2. Quick Start For Teammates

On a fresh Chameleon VM, use the root bootstrap script after cloning:

```bash
git clone <repo-url>
cd actual
git switch serving
bash bootstrap_chameleon.sh
```

The Chameleon bootstrap installs Docker/Compose, Node 22/Yarn, Python runtime
dependencies, starts serving/Prometheus/Grafana, generates realistic test
traffic, starts Actual, and prints the local and external service URLs.

Run all commands from the repository root.

```bash
git fetch origin
git switch serving
git pull origin serving
yarn install
```

Start the local final-project stack:

```bash
bash scripts/final_project_test.sh
```

The script generates test data, starts serving and monitoring services when
Docker Compose is available, sends realistic `/predict_batch` traffic, posts
data quality results, runs a promotion/rollback simulation, starts Actual if
needed, and opens the main service pages.

Important generated files:

```text
artifacts/test-data/actual_bank_transactions.qif
artifacts/test-data/actual_bank_transactions.csv
artifacts/test-data/synthetic_training_transactions.csv
serving/runtime/batch_input_realistic.json
```

Use the QIF file for Actual's account-level bank transaction import flow. The
CSV file is useful for inspecting data, but the Actual UI path shown in testing
uses QIF.

## Serving Folder Map

The `serving/` folder is now organized around the runtime modules that are
needed for the project demo and Chameleon deployment. Generated files are
ignored so the folder does not fill up with local run output.

| Path | Keep because | Main contents |
| --- | --- | --- |
| `app/` | Online inference service | FastAPI app, runtime state helpers, schemas, feature adapter, telemetry, model backends |
| `models/` | Required model artifacts | Source sklearn model, ONNX model, dynamic-quantized ONNX model, artifact manifest |
| `monitoring/` | Serving observability | Prometheus scrape/alert rules, Grafana dashboards and provisioning |
| `tools/` | Serving operations helpers | Artifact preparation, benchmarks, rollout action runner, packaging utilities |
| `tests/` | Serving correctness checks | API contract and feature adapter tests |
| `artifacts/examples/` | Stable API examples | Minimal single/batch input and output samples used by smoke checks |
| `results/summary/` | Checked-in evidence | Small benchmark and observability snapshots for reports/demos |
| `docker/` | Container build config | Serving Dockerfile |
| `docker-compose.yml` | Local service stack | Serving, Prometheus, Grafana, and tooling containers |
| `run.py` | Local orchestration entrypoint | Build, prepare, start, smoke, benchmark, monitor, and package commands |
| `bootstrap_chameleon.sh` | Serving-only compatibility wrapper | Delegates to the root Chameleon bootstrap with `--serving-only` |
| `README.md` | Single human entrypoint | System flow, project requirement mapping, services, tests, deployment notes |

Removed or ignored generated paths:

- `serving/.env`: local copy generated from `.env.example`
- `serving/runtime/`: request, prediction, feedback, and data-quality event logs
- `serving/.pytest_cache/`, `serving/**/__pycache__/`, `serving/**/*.pyc`: Python caches
- `serving/artifacts/gradescope/`: generated by `python3 run.py package`
- `serving/results/raw/`: raw benchmark JSONL output
- `artifacts/test-data/`, `artifacts/test-rollout/`, and `artifacts/chameleon/`: generated by bootstrap/test scripts
- `.venv-chameleon/`: local Python virtualenv created by `bootstrap_chameleon.sh`

The old standalone `serving/k8s/` handoff has been removed. The canonical
Kubernetes deployment is the integrated project deployment under:

```text
k8s/ml-system/base/
k8s/ml-system/overlays/staging/
k8s/ml-system/overlays/canary/
k8s/ml-system/overlays/production/
```

## 3. Local Service URLs

| Service | URL | Purpose |
| --- | --- | --- |
| Actual frontend | `http://127.0.0.1:3001` | Main user interface |
| Actual server-dev / sync server | `http://127.0.0.1:5006` | Local Actual server features |
| Serving API docs | `http://127.0.0.1:8000/docs` | API contract and manual requests |
| Serving monitor summary | `http://127.0.0.1:8000/monitor/summary` | Current model, request, feedback, and data quality behavior |
| Serving rollout decision | `http://127.0.0.1:8000/monitor/decision` | Promotion or rollback recommendation |
| Prometheus | `http://127.0.0.1:9090` | Metrics query UI |
| Grafana overview | `http://127.0.0.1:3000/d/actual-ml-system-overview/actual-ml-system-overview` | Main monitoring dashboard |
| Grafana home | `http://127.0.0.1:3000` | Dashboard index, login `admin / admin` |
| MLflow | `http://127.0.0.1:5000` | Training run tracking and model registry |
| MinIO console | `http://127.0.0.1:9001` | MLflow artifact storage UI |

Port difference:

- `3001` is the Actual web frontend. This is where users import and review transactions.
- `5006` is the Actual server-dev/sync server. It is not the main UI for bank transaction import.
- `8000` is the smart categorization serving API.
- `9090` and `3000` are monitoring services.
- `5000` is MLflow for experiment tracking/model registry.
- `9001` is MinIO for MLflow artifact inspection.

## 4. Final Defense Test Order

Use this sequence for a clean demo or team walkthrough.

### Step 1: Start Everything

```bash
bash scripts/final_project_test.sh
```

Confirm these pages open or are reachable:

```bash
curl -s http://127.0.0.1:8000/readyz
curl -s http://127.0.0.1:8000/monitor/summary
curl -s http://127.0.0.1:8000/monitor/decision
```

### Step 2: Test The Actual User Flow

1. Open `http://127.0.0.1:3001`.
2. On the file manager page, choose `Create test file`.
3. Do not use `Import file` on the manager page. That imports an entire Actual budget file, not bank transactions.
4. Open or create a checking account.
5. In the account page toolbar, select `Import`.
6. Import this file:

```text
artifacts/test-data/actual_bank_transactions.qif
```

7. Open the account transaction table.
8. Confirm the table includes:

```text
Category
AI Suggestion
All Top-3
Payment
Deposit
```

Expected behavior:

- `AI Suggestion` shows the selected predicted category and confidence score.
- `All Top-3` shows three predicted candidates and scores.
- The highest-confidence option is highlighted by default.
- Clicking another Top-3 option updates the highlight and the transaction category.
- The click is recorded as feedback for monitoring and retraining.

### Step 3: Test Serving Directly

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

### Step 4: Check Monitoring

Prometheus queries:

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
smartcat:request_rate_5m
smartcat:error_rate_5m
smartcat:p95_latency_ms_5m
smartcat:top1_acceptance_1h
smartcat:top3_acceptance_1h
```

Grafana dashboard:

```text
http://127.0.0.1:3000/d/actual-ml-system-overview/actual-ml-system-overview
```

The dashboard should show service health, request volume, latency, error rate,
predicted class distribution, confidence, feedback acceptance, data quality, and
rollout alerts.

### Step 5: Test Data Quality

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

### Step 6: Test Promotion And Rollback

Simulation without requiring a full training run:

```bash
python3 scripts/simulate_promotion_rollback.py
cat artifacts/test-rollout/archive/last_decision.json | jq
```

Dry-run the serving-owned rollout trigger:

```bash
python3 serving/tools/execute_rollout_action.py \
  --monitor-url http://127.0.0.1:8000/monitor/decision
```

Execute the recommended action only when appropriate:

```bash
python3 serving/tools/execute_rollout_action.py \
  --execute \
  --monitor-url http://127.0.0.1:8000/monitor/decision \
  --reload-url http://127.0.0.1:8000/admin/reload-model
```

### Step 7: Run A Compressed One-Week Automation Test

The final production submission runs under emulated user traffic for about one
week. For local testing, use a compressed week where one simulated hour takes
one second:

```bash
python3 scripts/run_week_simulation.py \
  --serving-url http://127.0.0.1:8000 \
  --simulated-hours 168 \
  --seconds-per-hour 1 \
  --pipeline-every-hours 24 \
  --rollout-every-hours 6
```

For a real-time week, use:

```bash
python3 scripts/run_week_simulation.py --seconds-per-hour 3600
```

What this does:

- Sends realistic `/predict_batch` traffic.
- Posts simulated user feedback to `/feedback`.
- Triggers `scripts/run_mlops_pipeline.py` on schedule.
- Registers challenger models in MLflow when `MLFLOW_TRACKING_URI` is set.
- Runs the serving-owned rollout trigger so a good model can replace the deployed model automatically.
- Calls `/admin/reload-model` after successful promotion or rollback.

Evidence is written to:

```text
artifacts/week-simulation/events.jsonl
artifacts/archive/last_decision.json
```

## 5. Service Inventory

| Area | Service or job | Local port or schedule | Responsibility | Main files |
| --- | --- | --- | --- | --- |
| Product UI | Actual frontend | `3001` | Import transactions and show predictions in the account table | `packages/desktop-client/src/components/accounts/Account.tsx`, `packages/desktop-client/src/components/transactions/TransactionsTable.tsx`, `packages/desktop-client/src/components/Titlebar.tsx` |
| Product backend | Actual server-dev | `5006` | Local Actual server features used by the frontend flow | `packages/sync-server/src/app.ts`, root `package.json` scripts |
| Actual ML bridge | Core prediction/feedback handlers | internal | Calls serving and persists prediction/feedback records | `packages/loot-core/src/server/ml/app.ts`, `packages/loot-core/src/server/ml/service.ts`, `packages/loot-core/src/server/ml/store.ts`, `packages/loot-core/src/server/transactions/app.ts` |
| Inference | Serving API | `8000` | Serves prediction, feedback, monitoring, and rollout endpoints | `serving/app/main.py`, `serving/app/runtime.py`, `serving/app/schemas.py`, `serving/app/config.py` |
| Model runtime | Runtime backends | loaded by serving | Loads sklearn, ONNX, or dynamic-quantized ONNX artifacts | `serving/app/backends/`, `serving/models/source/`, `serving/models/optimized/`, `serving/models/manifest.json` |
| Metrics | Prometheus | `9090` | Scrapes metrics and evaluates alert/recording rules | `serving/monitoring/prometheus.yml`, `serving/monitoring/prometheus-alerts.yml`, `serving/docker-compose.yml` |
| Dashboards | Grafana | `3000` | Visualizes service, prediction, feedback, data quality, and rollout state | `serving/monitoring/grafana/provisioning/`, `serving/monitoring/grafana/dashboards/`, `serving/docker-compose.yml` |
| Model registry | MLflow + Postgres + MinIO | `5000`, `9000`, `9001` | Tracks training runs, stores artifacts, and records promoted model aliases | `serving/docker-compose.yml`, `k8s/ml-system/base/mlflow-platform.yaml`, `training/train_model.py`, `scripts/promote_model.py`, `scripts/rollback_model.py` |
| Data quality | Data quality monitor | local command / K8s every 30 minutes | Checks ingestion, training set, and online drift quality | `data/data_quality_check.py`, `data/ingest.py`, `data/batch_pipeline.py`, `k8s/ml-system/base/data-quality-cronjob.yaml` |
| Training automation | Retrain/evaluate/promote pipeline | local command / CI / K8s every 6-12 hours | Builds datasets, trains challengers, evaluates gates, promotes winners | `scripts/run_mlops_pipeline.py`, `training/build_training_set.py`, `training/train_model.py`, `scripts/promote_model.py`, `.github/workflows/mlops-automation.yml`, `k8s/ml-system/base/training-pipeline-cronjob.yaml` |
| Rollout control | Serving rollout decision job | local command / K8s every 15 minutes | Runs promotion or rollback actions from monitoring decisions | `serving/tools/execute_rollout_action.py`, `scripts/promote_model.py`, `scripts/rollback_model.py`, `k8s/ml-system/base/rollout-decision-cronjob.yaml` |
| Deployment | Kubernetes manifests | staging/canary/production | Runs integrated services and automation on Chameleon | `k8s/ml-system/base/`, `k8s/ml-system/overlays/staging/`, `k8s/ml-system/overlays/canary/`, `k8s/ml-system/overlays/production/` |
| Test orchestration | Final project test runner | local script | Generates data, starts services, sends traffic, posts quality metrics, opens UIs | `scripts/final_project_test.sh`, `scripts/generate_actual_bank_import.py`, `scripts/simulate_promotion_rollback.py` |

## 6. Module Details

### 6.1 Actual Frontend

Purpose:

- Let users import bank transactions into an account.
- Request predictions for visible account transactions.
- Display `AI Suggestion` and `All Top-3` columns.
- Let users choose any Top-3 candidate.
- Update the transaction category after a user choice.

Main files:

```text
packages/desktop-client/src/components/accounts/Account.tsx
packages/desktop-client/src/components/transactions/TransactionsTable.tsx
packages/desktop-client/src/components/Titlebar.tsx
```

Behavior:

- `Account.tsx` builds prediction payloads from transaction fields and calls Actual core methods.
- `TransactionsTable.tsx` renders the prediction columns and user-selectable candidates.
- `Titlebar.tsx` displays category review status without making the UI look like an error.

### 6.2 Actual Core Bridge

Purpose:

- Provide frontend-callable ML methods.
- Call serving `/predict` and `/predict_batch`.
- Store predictions in local SQLite tables.
- Record feedback when a user chooses a category.
- Mirror feedback to serving so monitoring has real acceptance signals.

Main files:

```text
packages/loot-core/src/server/ml/app.ts
packages/loot-core/src/server/ml/service.ts
packages/loot-core/src/server/ml/store.ts
packages/loot-core/src/server/transactions/app.ts
```

Important methods:

```text
ml-predict-category
ml-predict-category-batch
ml-get-latest-prediction
ml-record-feedback
```

### 6.3 Serving API

Purpose:

- Own online model inference.
- Own feedback ingestion for monitoring.
- Own model behavior summaries and rollout decisions.
- Expose Prometheus metrics.
- Reload the active model after promotion or rollback.

Main files:

```text
serving/app/main.py
serving/app/runtime.py
serving/app/schemas.py
serving/app/config.py
serving/app/feature_adapter.py
serving/app/backends/
```

Main endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | Single transaction category prediction |
| `POST /predict_batch` | Batch category prediction for account tables/imports |
| `POST /feedback` | User-selected category feedback |
| `GET /metrics` | Prometheus scrape endpoint |
| `GET /monitor/summary` | Aggregated request, prediction, feedback, and data quality summary |
| `GET /monitor/decision` | Promotion/rollback recommendation based on thresholds |
| `POST /monitor/data-quality` | Receives data quality check results |
| `POST /admin/reload-model` | Clears backend cache and reloads deployed model after promotion/rollback |
| `GET /healthz` | Liveness health check |
| `GET /readyz` | Readiness check including model warmup readiness |
| `GET /versionz` | Model/runtime version info |

Key environment variables:

```text
BACKEND_KIND
MODEL_PATH
SOURCE_MODEL_PATH
MODEL_VERSION
TOP_K
ROLLOUT_CONTEXT
PROMOTION_MIN_REQUESTS
PROMOTION_MIN_FEEDBACK
PROMOTION_MIN_TOP1_ACCEPTANCE
PROMOTION_MIN_TOP3_ACCEPTANCE
ROLLBACK_MIN_REQUESTS
ROLLBACK_MIN_FEEDBACK
ROLLBACK_MAX_P95_MS
ROLLBACK_MAX_ERROR_RATE
ROLLBACK_MIN_TOP1_ACCEPTANCE
ROLLBACK_MIN_TOP3_ACCEPTANCE
```

Defined in:

```text
serving/.env.example
serving/app/config.py
serving/docker-compose.yml
k8s/ml-system/base/kustomization.yaml
```

### 6.4 Model Artifacts

Purpose:

- Provide model files loaded by serving.
- Keep source and optimized model variants together.
- Document labels and artifact metadata.

Main files:

```text
serving/models/source/v2_tfidf_linearsvc_model.joblib
serving/models/optimized/v2_tfidf_linearsvc_model.onnx
serving/models/optimized/v2_tfidf_linearsvc_model.dynamic_quant.onnx
serving/models/manifest.json
serving/tools/prepare_artifacts.py
serving/tools/benchmark_artifacts.py
```

Runtime variants:

| Variant | `BACKEND_KIND` | Model path |
| --- | --- | --- |
| sklearn baseline | `baseline` | `serving/models/source/v2_tfidf_linearsvc_model.joblib` |
| ONNX | `onnx` | `serving/models/optimized/v2_tfidf_linearsvc_model.onnx` |
| ONNX dynamic quantization | `onnx_dynamic_quant` | `serving/models/optimized/v2_tfidf_linearsvc_model.dynamic_quant.onnx` |

### 6.5 Monitoring

Prometheus purpose:

- Scrape serving `/metrics`.
- Store time-series metrics.
- Evaluate alert rules.
- Define summary recording rules used by Grafana.

Main files:

```text
serving/monitoring/prometheus.yml
serving/monitoring/prometheus-alerts.yml
serving/docker-compose.yml
```

Important metrics:

```text
http_requests_total
http_request_duration_seconds
prediction_confidence
predicted_class_total
feedback_total
feedback_match_total
feedback_top3_match_total
actual_data_quality_pass
actual_data_quality_issue_count
actual_data_quality_metric
```

Recording rules:

```text
smartcat:request_rate_5m
smartcat:error_rate_5m
smartcat:p95_latency_ms_5m
smartcat:p50_latency_ms_5m
smartcat:avg_confidence_5m
smartcat:low_confidence_ratio_5m
smartcat:top1_acceptance_1h
smartcat:top3_acceptance_1h
smartcat:data_quality_pass_min
```

Alert rules:

```text
SmartcatServingHighP95Latency
SmartcatServingHighErrorRate
SmartcatServingLowTop1Acceptance
SmartcatServingLowTop3Acceptance
SmartcatCandidatePromotionReady
SmartcatDataQualityGateFailed
SmartcatOnlineDriftHigh
```

Grafana purpose:

- Show service health, model output, user feedback, data quality, and rollout status.
- Provide one overview page for demos and production checks.

Main files:

```text
serving/monitoring/grafana/provisioning/datasources/datasource.yml
serving/monitoring/grafana/provisioning/dashboards/dashboards.yml
serving/monitoring/grafana/dashboards/system_overview.json
serving/monitoring/grafana/dashboards/service_monitoring.json
serving/monitoring/grafana/dashboards/prediction_monitoring.json
serving/monitoring/grafana/dashboards/feedback_monitoring.json
serving/monitoring/grafana/dashboards/data_quality_monitoring.json
```

### 6.6 Data Quality

Purpose:

- Evaluate data quality at ingestion.
- Evaluate training-set quality before retraining.
- Monitor online inference drift.
- Publish data quality status into serving, Prometheus, and Grafana.

Main files:

```text
data/data_quality_check.py
data/ingest.py
data/batch_pipeline.py
data/online_features.py
data/feedback_collector.py
k8s/ml-system/base/data-quality-cronjob.yaml
```

Checks published to serving:

```text
ingestion
training_set
online_drift
```

Local example:

```bash
python3 data/data_quality_check.py ingestion \
  --input artifacts/test-data/synthetic_training_transactions.csv \
  --output-json artifacts/test-data/ingestion_quality.json \
  --post-url http://127.0.0.1:8000/monitor/data-quality
```

Kubernetes schedule:

```text
CronJob: data-quality-monitor
Schedule: */30 * * * *
File: k8s/ml-system/base/data-quality-cronjob.yaml
```

### 6.7 Training, Promotion, And Rollback

Purpose:

- Compile training data.
- Run data quality gates.
- Train a challenger model.
- Evaluate task-specific metrics.
- Register the challenger in MLflow when a tracking URI is configured.
- Promote the challenger if it passes thresholds and is at least as good as the current deployed model.
- Update the MLflow production/canary/staging alias after promotion.
- Archive rollback metadata.
- Roll back the active model and restore the MLflow alias if production metrics degrade.

Main files:

```text
training/build_training_set.py
training/train_model.py
scripts/run_mlops_pipeline.py
scripts/promote_model.py
scripts/rollback_model.py
scripts/simulate_promotion_rollback.py
scripts/run_week_simulation.py
.github/workflows/mlops-automation.yml
docker/mlops.Dockerfile
k8s/ml-system/base/mlflow-platform.yaml
k8s/ml-system/base/training-pipeline-cronjob.yaml
k8s/ml-system/base/rollout-decision-cronjob.yaml
serving/tools/execute_rollout_action.py
```

Local automated pipeline:

```bash
python3 scripts/run_mlops_pipeline.py --synthetic-bootstrap
```

CI automation:

```text
Workflow: Smart Transaction MLOps Automation
File: .github/workflows/mlops-automation.yml
Schedule: 0 */6 * * *
```

Kubernetes automation:

```text
CronJob: ml-retrain-evaluate-promote
Base schedule: 0 */6 * * *
File: k8s/ml-system/base/training-pipeline-cronjob.yaml

CronJob: serving-rollout-decision
Schedule: */15 * * * *
File: k8s/ml-system/base/rollout-decision-cronjob.yaml
```

Promotion gates:

```text
minimum Top-3 accuracy
minimum macro F1
challenger metric must be at least as good as current deployed metric
```

Default thresholds:

```text
--min-top3-accuracy 0.70
--min-macro-f1 0.55
```

Rollback triggers:

```text
p95 latency above threshold
5xx error rate above threshold
Top-1 acceptance below threshold
Top-3 acceptance below threshold
```

Decision source:

```text
GET /monitor/decision
```

Possible decision actions:

| Action | Meaning |
| --- | --- |
| `hold` | No action required |
| `promote_candidate` | Candidate has enough traffic, feedback, latency, error-rate, and acceptance evidence for promotion |
| `rollback_active` | Production behavior crossed rollback thresholds |

## 7. Project Requirement Traceability

### Joint Requirement: Integrated System Running On Chameleon

Implementation:

- The system has Kubernetes manifests for Actual server, serving, Prometheus, Grafana, persistent storage, and automation CronJobs.
- Staging, canary, and production are represented as separate namespaces/overlays.
- For Chameleon, use one Kubernetes cluster with multiple services and namespaces, not four separate clusters. The service split should be by function, while environments are separated by namespace.

Evidence files:

```text
k8s/ml-system/base/
k8s/ml-system/overlays/staging/
k8s/ml-system/overlays/canary/
k8s/ml-system/overlays/production/
```

How to test:

```bash
kubectl kustomize k8s/ml-system/overlays/staging
kubectl kustomize k8s/ml-system/overlays/canary
kubectl kustomize k8s/ml-system/overlays/production
```

### Joint Requirement: End-To-End Plumbing

Implementation:

- Actual imports transactions and stores them in SQLite.
- Account UI requests `/predict_batch` through Actual core.
- Serving returns Top-3 predictions.
- User choice is stored as feedback.
- Data quality, training, evaluation, promotion, deployment, and rollback scripts connect the lifecycle.

Evidence files:

```text
packages/desktop-client/src/components/accounts/Account.tsx
packages/desktop-client/src/components/transactions/TransactionsTable.tsx
packages/loot-core/src/server/ml/
serving/app/main.py
data/data_quality_check.py
training/build_training_set.py
training/train_model.py
scripts/run_mlops_pipeline.py
scripts/promote_model.py
scripts/rollback_model.py
```

How to test:

```bash
bash scripts/final_project_test.sh
```

Then import:

```text
artifacts/test-data/actual_bank_transactions.qif
```

### Joint Requirement: Feature Implemented Inside The Open Source Service

Implementation:

- The prediction feature appears inside Actual's normal account transaction table.
- The user does not need a separate external demo page.

Evidence files:

```text
packages/desktop-client/src/components/accounts/Account.tsx
packages/desktop-client/src/components/transactions/TransactionsTable.tsx
```

How to test:

- Open Actual on `3001`.
- Create a test file.
- Import QIF transactions inside an account.
- Confirm `AI Suggestion` and `All Top-3` appear in the transaction table.

### Joint Requirement: Automation With Minimal Human Work

Implementation:

- Local script starts the demo stack and sends traffic.
- CI workflow runs the MLOps pipeline on a schedule.
- K8s CronJobs run data quality, retraining/promotion, and rollout decisions.
- Promotion and rollback scripts are executable without manually editing files.

Evidence files:

```text
scripts/final_project_test.sh
scripts/run_mlops_pipeline.py
.github/workflows/mlops-automation.yml
k8s/ml-system/base/data-quality-cronjob.yaml
k8s/ml-system/base/training-pipeline-cronjob.yaml
k8s/ml-system/base/rollout-decision-cronjob.yaml
serving/tools/execute_rollout_action.py
```

How to test locally:

```bash
python3 scripts/simulate_promotion_rollback.py
python3 serving/tools/execute_rollout_action.py --monitor-url http://127.0.0.1:8000/monitor/decision
```

### Joint Requirement: Safeguarding Plan

Implementation:

- Fairness: monitor quality by available data segments such as country and currency.
- Privacy: serving does not persist raw transaction descriptions in prediction event logs.
- Robustness: batch prediction has fallback paths, health checks, and rollback triggers.
- Explainability and transparency: users see Top-3 categories and confidence scores.
- Accountability: user corrections are recorded as feedback and promotion/rollback decisions are archived.
- Human control: users can override predictions with one click.

Evidence files:

```text
docs/ml-system/SAFEGUARDING.md
data/data_quality_check.py
serving/app/main.py
serving/tools/execute_rollout_action.py
packages/desktop-client/src/components/transactions/TransactionsTable.tsx
packages/loot-core/src/server/ml/store.ts
scripts/promote_model.py
scripts/rollback_model.py
```

How to test:

- Click a non-highlighted Top-3 candidate in Actual.
- Confirm the category changes.
- Confirm `/monitor/summary` shows feedback activity.
- Confirm Prometheus has feedback metrics.

### Training Role Requirement: Meaningful Evaluation And Quality Gates

Implementation:

- Training creates train/test splits.
- Training records evaluation metrics.
- Promotion only happens if the challenger passes Top-3 accuracy and macro-F1 gates.

Evidence files:

```text
training/build_training_set.py
training/train_model.py
scripts/run_mlops_pipeline.py
scripts/promote_model.py
```

How to test:

```bash
python3 scripts/run_mlops_pipeline.py --synthetic-bootstrap
```

### Serving Role Requirement: Monitoring And Promotion/Rollback Triggers

Implementation:

- Serving monitors model output, operational metrics, and user feedback.
- `/monitor/decision` converts monitoring state into `hold`, `promote_candidate`, or `rollback_active`.
- `execute_rollout_action.py` executes the serving-owned trigger path.

Evidence files:

```text
serving/app/main.py
serving/tools/execute_rollout_action.py
serving/monitoring/prometheus-alerts.yml
serving/monitoring/grafana/dashboards/system_overview.json
scripts/promote_model.py
scripts/rollback_model.py
```

How to test:

```bash
curl -s http://127.0.0.1:8000/monitor/decision | jq
python3 serving/tools/execute_rollout_action.py --monitor-url http://127.0.0.1:8000/monitor/decision
```

### Data Role Requirement: Three-Point Data Quality Monitoring

Implementation:

- Ingestion checks validate incoming/raw data.
- Training-set checks validate compiled training data before retraining.
- Online drift checks monitor live inference data.
- Results are posted to serving and exposed through Prometheus/Grafana.

Evidence files:

```text
data/data_quality_check.py
data/ingest.py
data/batch_pipeline.py
data/online_features.py
serving/app/main.py
serving/monitoring/grafana/dashboards/data_quality_monitoring.json
k8s/ml-system/base/data-quality-cronjob.yaml
```

How to test:

```bash
python3 data/data_quality_check.py ingestion \
  --input artifacts/test-data/synthetic_training_transactions.csv \
  --output-json artifacts/test-data/ingestion_quality.json \
  --post-url http://127.0.0.1:8000/monitor/data-quality
```

### DevOps/Platform Requirement: Infrastructure Monitoring And Scaling

Implementation:

- Prometheus and Grafana monitor service health and performance.
- K8s manifests include readiness/liveness checks and an HPA for serving.
- Alerts capture latency, error rate, acceptance drops, and data quality failures.

Evidence files:

```text
serving/monitoring/prometheus.yml
serving/monitoring/prometheus-alerts.yml
serving/monitoring/grafana/dashboards/system_overview.json
k8s/ml-system/base/serving.yaml
k8s/ml-system/base/monitoring.yaml
```

How to test:

```bash
docker run --rm --entrypoint promtool \
  -v "$PWD/serving/monitoring:/etc/prometheus:ro" \
  prom/prometheus:latest check config /etc/prometheus/prometheus.yml
```

## 8. Kubernetes And Chameleon Deployment

For the final project, the recommended design is one Kubernetes cluster on
Chameleon with multiple services and namespaces:

```text
One Chameleon K8s cluster
        |
        +-- actual-ml-staging namespace
        +-- actual-ml-canary namespace
        +-- actual-ml-production namespace
```

Do not create four separate Kubernetes clusters for four team roles. Use one
integrated cluster because the course requirement asks for a unified system,
not role-owned duplicated infrastructure.

Base services:

| Service | File | Port |
| --- | --- | --- |
| `actual-sync` | `k8s/ml-system/base/actual-sync.yaml` | `5006` |
| `smartcat-serving` | `k8s/ml-system/base/serving.yaml` | `8000` |
| `prometheus` | `k8s/ml-system/base/monitoring.yaml` | `9090` |
| `grafana` | `k8s/ml-system/base/monitoring.yaml` | `3000` |

Environment overlays:

```text
k8s/ml-system/overlays/staging/
k8s/ml-system/overlays/canary/
k8s/ml-system/overlays/production/
```

Deploy:

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

Storage:

```text
actual-data PVC: Actual data
ml-artifacts PVC: model artifacts, pipeline outputs, deployment archives
monitoring-data PVC: Prometheus storage
```

Scaling:

```text
HorizontalPodAutoscaler: smartcat-serving
Base replicas: 2
Base min/max: 2/6
Metric: CPU utilization 70%
```

Local vs Chameleon differences:

| Topic | Local / Docker Compose | Chameleon / Kubernetes |
| --- | --- | --- |
| Service discovery | `localhost` ports | Kubernetes service DNS and namespaces |
| Storage | local folders and Docker volumes | PVCs and Chameleon storage classes |
| Images | built locally | images should be pushed to a registry or loaded into cluster nodes |
| Environments | one local stack | staging, canary, and production overlays |
| Automation | scripts and optional local commands | CronJobs and CI workflow |
| Access | browser on `127.0.0.1` | NodePort, LoadBalancer, Ingress, or port-forward |
| Monitoring data | local Prometheus volume | persistent monitoring PVC |

Before a real Chameleon run, confirm image names, storage classes, secrets, and
external access match the team's Chameleon setup.

## 9. Docker Compose Local Stack

Purpose:

- Run serving and monitoring locally without Kubernetes.
- Mount dashboard and Prometheus configs directly from the repository.

Main file:

```text
serving/docker-compose.yml
```

Services:

| Compose service | Container | Purpose |
| --- | --- | --- |
| `serve` | `actualbudget-serving-app` | FastAPI serving service |
| `tooling` | profile `tools` | Utility container for Python commands |
| `prometheus` | `actualbudget-serving-prometheus` | Metrics and alerts |
| `grafana` | `actualbudget-serving-grafana` | Dashboards |

Start through helper:

```bash
cd serving
python3 run.py monitor-up
```

Manual serving workflow:

```bash
cd serving
python3 run.py doctor
python3 run.py build
python3 run.py prepare
python3 run.py up --variant onnx_dynamic_quant --workers 2
python3 run.py smoke
```

## 10. Data, Training, Serving, And Frontend Ownership

Data side:

- Provides ingestion and synthetic/test data utilities.
- Checks data quality at ingestion, training-set build time, and online drift.
- Sends data quality results to serving.

Main files:

```text
data/data_quality_check.py
data/ingest.py
data/batch_pipeline.py
data/online_features.py
data/feedback_collector.py
scripts/generate_actual_bank_import.py
```

Training side:

- Builds training datasets.
- Trains and evaluates challenger models.
- Runs quality gates before registration/promotion.

Main files:

```text
training/build_training_set.py
training/train_model.py
scripts/run_mlops_pipeline.py
scripts/promote_model.py
scripts/rollback_model.py
```

Serving side:

- Serves predictions.
- Tracks model output, operational metrics, and user feedback.
- Owns promotion/rollback trigger decisions.
- Exposes Prometheus metrics and Grafana dashboards.

Main files:

```text
serving/app/main.py
serving/app/runtime.py
serving/tools/execute_rollout_action.py
serving/monitoring/prometheus-alerts.yml
serving/monitoring/grafana/dashboards/
```

Frontend / product side:

- Integrates predictions into Actual's real transaction workflow.
- Shows Top-3 suggestions and lets users override predictions.
- Sends user corrections back as feedback.

Main files:

```text
packages/desktop-client/src/components/accounts/Account.tsx
packages/desktop-client/src/components/transactions/TransactionsTable.tsx
packages/loot-core/src/server/ml/
```

## 11. Input And Output Contracts

Single prediction request:

```json
{
  "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
  "country": "US",
  "currency": "USD",
  "amount": -6.45,
  "transaction_date": "2026-04-20",
  "account_name": "Checking",
  "notes": "coffee"
}
```

Batch prediction request:

```json
{
  "items": [
    {
      "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
      "country": "US",
      "currency": "USD",
      "amount": -6.45,
      "transaction_date": "2026-04-20"
    }
  ]
}
```

Prediction response includes:

```text
predicted_category_id
confidence
top_categories
model_version
```

The current model consumes transaction text plus country/currency context. Extra
metadata fields are accepted so the feature contract can grow without breaking
frontend or data pipelines.

## 12. Common Mistakes

- `Import file` on the manager page imports an entire Actual budget file.
- Bank transactions must be imported from inside an account page.
- Use the generated QIF file for the account transaction import path.
- `3001` is the Actual frontend.
- `5006` is the Actual server-dev/sync server.
- `8000` is serving.
- `9090` is Prometheus.
- `3000` is Grafana.
- Seeing Top-3 predictions in the Actual table confirms frontend, Actual core, and serving are connected.
- Seeing feedback/data-quality metrics in Prometheus confirms monitoring is connected.

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
- K8s manifests render for staging, canary, and production.

## 14. Pre-Demo Validation Checklist

Check serving health:

```bash
curl -s http://127.0.0.1:8000/readyz
```

Check metrics:

```bash
curl -s http://127.0.0.1:8000/metrics | head
```

Check Prometheus config:

```bash
docker run --rm --entrypoint promtool \
  -v "$PWD/serving/monitoring:/etc/prometheus:ro" \
  prom/prometheus:latest check config /etc/prometheus/prometheus.yml
```

Check Grafana dashboard JSON:

```bash
python3 -m json.tool serving/monitoring/grafana/dashboards/system_overview.json >/tmp/system_overview.json
```

Check K8s manifests:

```bash
kubectl kustomize k8s/ml-system/overlays/staging
kubectl kustomize k8s/ml-system/overlays/canary
kubectl kustomize k8s/ml-system/overlays/production
```

## 15. Current Coverage And Remaining Gaps

This section is the short rubric status snapshot for the team. It separates
what is already implemented in this branch from what still needs deployment
evidence or small follow-up work before the final Chameleon run.

### Course Lab Alignment

This branch follows the same system shape as the course labs, but it currently
uses a lighter implementation for some platform pieces. The table below maps
the course lab pattern to this repository so the team can explain which parts
are already equivalent and which parts should be upgraded if time allows.

| Course lab pattern | What the lab emphasizes | Current project mapping | Status |
| --- | --- | --- | --- |
| MLOps Pipeline on Chameleon | `tf/`, `ansible/`, `k8s/platform`, `k8s/staging`, `k8s/canary`, `k8s/production`, and `workflows/` for lifecycle automation | We have `bootstrap_chameleon.sh`, `scripts/chameleon_bootstrap.sh`, `k8s/ml-system/base`, and `k8s/ml-system/overlays/staging|canary|production` | Functionally close, but not full course-style IaC/GitOps |
| MLOps lifecycle workflows | Training builds a candidate, deploys to staging, tests staging, promotes to canary, then promotes or rolls back production | `scripts/run_mlops_pipeline.py`, `scripts/promote_model.py`, `scripts/rollback_model.py`, `serving/tools/execute_rollout_action.py`, K8s CronJobs | Implemented as Python scripts, GitHub Actions, and CronJobs instead of Argo Workflows |
| MLflow tracking and registry | MLflow runs as shared platform service with Postgres metadata and MinIO/S3 artifacts | `serving/docker-compose.yml` and `k8s/ml-system/base/mlflow-platform.yaml` run MLflow/Postgres/MinIO; `training/train_model.py` registers challengers; promotion/rollback scripts update aliases | Implemented; final Chameleon run evidence still needed |
| Serving system lab | FastAPI inference service exposes `/predict`, batch-style serving, health checks, model artifact loading, and performance testing | `serving/app/main.py`, `serving/app/backends/`, `serving/tools/benchmark_http.py`, `serving/tools/benchmark_arrivals.py` | Implemented |
| Online evaluation lab | FastAPI exposes `/metrics`; Prometheus scrapes it; Grafana shows operational and prediction behavior | `serving/app/main.py`, `serving/monitoring/prometheus.yml`, `serving/monitoring/prometheus-alerts.yml`, Grafana dashboards | Implemented locally; K8s should mount the same dashboard/rule configs |
| Feedback-loop lab | Save production predictions, collect explicit user feedback, sample low-confidence or flagged items, and feed labels into retraining | Actual Top-3 clicks record feedback in `ml_feedback` and serving `/feedback`; `data/feedback_collector.py` and `training/build_training_set.py` support the retraining path | Implemented for user feedback; random/low-confidence sampling is documented as a possible extension |
| Data quality and drift | Check data at ingestion, training-set build time, and online production drift | `data/data_quality_check.py` plus `k8s/ml-system/base/data-quality-cronjob.yaml` | Implemented, but final Chameleon PVC paths must be verified with real pipeline output |
| Platform services | Shared platform services should avoid duplicated infrastructure across roles | Local Compose has one serving/Prometheus/Grafana stack; K8s overlays currently instantiate monitoring per environment | Acceptable for local demo; final K8s should ideally split shared `platform` namespace from app environments |

Practical refactor decision:

- Do not rewrite the project into full Terraform/Ansible/Argo tonight; it is a separate platform migration.
- Keep the current working Compose/Kustomize/CronJob system as the demo path.
- If the team has more time, migrate the K8s layout toward the course `platform + staging + canary + production + workflows` shape.
- The highest-value upgrade is adding a shared platform namespace with MLflow, Postgres, MinIO, Prometheus, Grafana, and traffic routing.

### Already Implemented

| Requirement | Current implementation | Evidence |
| --- | --- | --- |
| Feature inside Actual user flow | Account tables show `AI Suggestion` and `All Top-3`; users can select any candidate and update the category | `packages/desktop-client/src/components/accounts/Account.tsx`, `packages/desktop-client/src/components/transactions/TransactionsTable.tsx` |
| Frontend to serving inference | Actual core calls `/predict_batch` first and supports `/predict` fallback | `packages/loot-core/src/server/ml/service.ts`, `packages/loot-core/src/server/ml/app.ts` |
| Feedback capture | User category choices are stored in `ml_feedback` and mirrored to serving `/feedback` | `packages/loot-core/src/server/ml/store.ts`, `serving/app/main.py` |
| Serving monitoring | Serving exposes model output metrics, latency/error metrics, feedback metrics, data-quality metrics, and `/monitor/summary` | `serving/app/main.py`, `serving/app/config.py` |
| Promotion/rollback decision endpoint | `/monitor/decision` returns `hold`, `promote_candidate`, or `rollback_active` using traffic, latency, error-rate, and feedback thresholds | `serving/app/main.py` |
| Rollout trigger runner | Serving-owned trigger reads `/monitor/decision`, runs promotion or rollback, and can reload the deployed model | `serving/tools/execute_rollout_action.py` |
| Training/evaluation gate | Challenger must pass Top-3 accuracy and macro-F1 gates and must not underperform the current champion | `training/train_model.py`, `scripts/run_mlops_pipeline.py`, `scripts/promote_model.py` |
| MLflow registry loop | Training registers challengers when `MLFLOW_TRACKING_URI` is configured; promotion/rollback updates the active MLflow alias | `training/train_model.py`, `scripts/promote_model.py`, `scripts/rollback_model.py`, `k8s/ml-system/base/mlflow-platform.yaml` |
| Week-long traffic automation | A compressed or real-time one-week simulation sends traffic, records feedback, runs retraining, and triggers rollout decisions | `scripts/run_week_simulation.py` |
| Data quality checks | Ingestion, training-set, and online-drift checks exist and can post status into serving | `data/data_quality_check.py`, `k8s/ml-system/base/data-quality-cronjob.yaml` |
| Local full-stack test | One script generates import data, starts serving/monitoring, sends traffic, posts data quality, simulates rollout, and starts Actual | `scripts/final_project_test.sh` |
| Fresh Chameleon bootstrap | One root script installs Docker, Node/Yarn, Python deps, starts services, and prints URLs | `bootstrap_chameleon.sh`, `scripts/chameleon_bootstrap.sh` |
| K8s environment split | Staging, canary, and production overlays exist | `k8s/ml-system/overlays/staging/`, `k8s/ml-system/overlays/canary/`, `k8s/ml-system/overlays/production/` |
| DevOps health/scaling | Serving has readiness/liveness probes and HPA; Prometheus/Grafana manifests exist | `k8s/ml-system/base/serving.yaml`, `k8s/ml-system/base/monitoring.yaml` |
| Safeguarding mechanisms | Top-3 transparency, user override, feedback accountability, privacy-aware serving logs, rollback robustness | `docs/ml-system/SAFEGUARDING.md` |

### Frequent Gated Promotion

The intended model update policy is frequent evaluation, gated promotion, and
fast rollback rather than blind frequent replacement.

Implemented cadence:

```text
staging: validates wiring with low thresholds
canary: retrain/evaluate every 3 hours
production: conservative retrain/evaluate every 12 hours
rollout decision: serving-owned trigger checks every 15 minutes
```

Evidence:

```text
k8s/ml-system/overlays/staging/kustomization.yaml
k8s/ml-system/overlays/canary/kustomization.yaml
k8s/ml-system/overlays/production/kustomization.yaml
k8s/ml-system/base/training-pipeline-cronjob.yaml
k8s/ml-system/base/rollout-decision-cronjob.yaml
serving/tools/execute_rollout_action.py
serving/app/main.py
```

Promotion requires:

```text
enough candidate request volume
enough candidate feedback volume
p95 latency below promotion threshold
error rate below promotion threshold
Top-1 acceptance above promotion threshold
Top-3 acceptance above promotion threshold
offline Top-3 accuracy >= 0.70
offline macro F1 >= 0.55
challenger metric >= champion metric
```

Rollback triggers:

```text
production p95 latency above threshold
production 5xx/error rate above threshold
production Top-1 acceptance below threshold
production Top-3 acceptance below threshold
```

### Remaining Gaps Before Final Freeze

| Gap | Why it matters | Recommended next step |
| --- | --- | --- |
| No full Terraform/Ansible infrastructure path yet | The course MLOps lab provisions Chameleon resources with Terraform and configures Kubernetes/Argo with Ansible; our bootstrap script is easier but less lab-like | Either add `tf/` and `ansible/` from the team's Chameleon setup, or explicitly document that the team is using a manual cluster plus bootstrap script |
| No Argo Workflows/ArgoCD lifecycle yet | The course lifecycle path uses Argo Workflows for train/build/deploy/test/promote and ArgoCD for environment sync | Convert the existing CronJobs and GitHub workflow into Argo `WorkflowTemplate`s if final grading expects lab-style GitOps evidence |
| Shared platform namespace is not separated yet | The course lab keeps accessory services in `k8s/platform` and app versions in staging/canary/production; our base repeats monitoring with each overlay | Add `k8s/ml-system/platform` for MLflow, MinIO/Postgres, Prometheus, Grafana, and gateway/traffic routing |
| K8s Actual image may not include this branch's frontend/core changes | The current K8s manifest references `actualbudget/actual-sync:latest`; final Kubernetes demo must run the modified Actual UI with prediction columns | Build and push a custom Actual image from this branch, then update `k8s/ml-system/base/actual-sync.yaml` or split web/sync services with the custom image |
| K8s Grafana/Prometheus config is lighter than local Compose | Local Compose has full dashboard provisioning and alert rules; K8s manifest currently has basic Prometheus/Grafana setup | Mount `serving/monitoring/prometheus-alerts.yml` and Grafana dashboard JSONs through K8s ConfigMaps |
| Training CronJob still uses synthetic bootstrap as a fallback path | Rubric prefers production data -> feedback -> retraining, not only synthetic data | Make `scripts/run_mlops_pipeline.py` prefer feedback-built datasets from `training/build_training_set.py`, and use synthetic data only when feedback volume is too small |
| MLflow registry needs Chameleon evidence | MLflow/Postgres/MinIO manifests and alias updates now exist, but final grading will want screenshots/logs from the running Chameleon deployment | Capture MLflow experiment runs, registered model versions, and `production`/`canary` alias changes during the week simulation |
| Chameleon K8s needs real run evidence | Manifests render, but final grading will expect the system running on Chameleon | Apply staging/canary/production overlays, capture pods/services/CronJobs/logs, and record Grafana/Prometheus/video evidence |
| Data-quality CronJob paths need production PVC data | The CronJob expects files under `/workspace/artifacts/data/`; those must exist in the shared PVC | Ensure pipeline writes canonical train/val/online feature files to those paths, or update the CronJob to use pipeline output paths |
| Alerting lacks notification routing | Prometheus rules exist, but there is no Alertmanager notification path in K8s | For final polish, add Alertmanager or document dashboard-based alert review if notification setup is out of scope |

### Recommended Final Evidence To Collect

Before the implementation freeze, collect these artifacts for the report/video:

```bash
kubectl get pods -A
kubectl get svc -A
kubectl get cronjobs -A
kubectl get jobs -n actual-ml-canary
kubectl get jobs -n actual-ml-production
kubectl logs -n actual-ml-canary job/<latest-ml-retrain-job-name> --tail=200
kubectl logs -n actual-ml-production job/<latest-rollout-decision-job-name> --tail=200
curl -s http://127.0.0.1:8000/monitor/summary | jq
curl -s http://127.0.0.1:8000/monitor/decision | jq
```

Also capture:

- Actual account table showing `Category`, `AI Suggestion`, and `All Top-3`.
- Grafana system overview with request rate, latency, error rate, confidence, feedback, and data-quality panels.
- Prometheus query results for `feedback_total`, `prediction_confidence`, and `actual_data_quality_pass`.
- `artifacts/archive/last_decision.json` or the K8s PVC equivalent after a promotion/rollback simulation.

## 16. What Needs Environment-Specific Setup On Chameleon

The repository contains the integrated code, manifests, and automation hooks.
Before a final Chameleon production-style run, the team still needs to confirm
these deployment-specific items:

- Container image registry and image tags used by K8s manifests.
- Chameleon storage class names for PVCs.
- External access method for Actual, serving, Prometheus, and Grafana.
- Secrets or environment variables that should not be hardcoded.
- Whether the final demo uses port-forward, NodePort, LoadBalancer, or Ingress.
- Whether automatic promotion should execute directly or require manual approval from canary to production.

These are deployment choices, not separate role-owned systems. Keep shared
infrastructure unified unless there is a clear technical reason to split it.
