# Smart Transaction Categorization Services

This document summarizes the services, automation jobs, monitoring stack, and
rollout controls used by the integrated smart transaction categorization system.
It is intended as a service inventory for team handoff, demos, and Chameleon
operation.

## Service Map

| Area | Service or job | Local port or schedule | Main responsibility | Primary files |
| --- | --- | --- | --- | --- |
| Product UI | Actual frontend | `3001` | User imports transactions and reviews category predictions in the normal Actual account table | `packages/desktop-client/src/components/accounts/Account.tsx`, `packages/desktop-client/src/components/transactions/TransactionsTable.tsx`, `packages/desktop-client/src/components/Titlebar.tsx` |
| Product backend | Actual sync/server-dev | `5006` | Runs Actual server features used by the local frontend flow | `packages/sync-server/src/app.ts`, `package.json` scripts |
| Inference | Serving API | `8000` | Serves `/predict`, `/predict_batch`, `/feedback`, `/monitor/*`, and `/metrics` | `serving/app/main.py`, `serving/app/runtime.py`, `serving/app/schemas.py`, `serving/app/config.py` |
| Model runtime | Model backends | loaded by serving | Loads sklearn, ONNX, or dynamic-quantized ONNX model artifacts | `serving/app/backends/`, `serving/models/source/`, `serving/models/optimized/`, `serving/models/manifest.json` |
| Metrics | Prometheus | `9090` | Scrapes serving metrics, evaluates alert rules, and exposes query UI | `serving/monitoring/prometheus.yml`, `serving/monitoring/prometheus-alerts.yml`, `serving/docker-compose.yml` |
| Dashboards | Grafana | `3000` | Shows service, prediction, feedback, data quality, and system overview dashboards | `serving/monitoring/grafana/provisioning/`, `serving/monitoring/grafana/dashboards/`, `serving/docker-compose.yml` |
| Data quality | Data quality monitor | local command / K8s every 30 minutes | Checks ingestion quality, training set quality, and online drift, then posts results to serving | `data/data_quality_check.py`, `data/ingest.py`, `data/batch_pipeline.py`, `k8s/ml-system/base/data-quality-cronjob.yaml` |
| Training automation | Retrain/evaluate/promote pipeline | local command / CI / K8s every 6-12 hours | Builds datasets, runs data quality gate, trains challenger, evaluates, and promotes if gates pass | `scripts/run_mlops_pipeline.py`, `training/build_training_set.py`, `training/train_model.py`, `scripts/promote_model.py`, `k8s/ml-system/base/training-pipeline-cronjob.yaml`, `.github/workflows/mlops-automation.yml` |
| Rollout control | Serving rollout decision job | local command / K8s every 15 minutes | Reads serving monitor decision and runs promotion or rollback action | `serving/tools/execute_rollout_action.py`, `scripts/promote_model.py`, `scripts/rollback_model.py`, `k8s/ml-system/base/rollout-decision-cronjob.yaml` |
| Deployment | Kubernetes manifests | staging/canary/production | Runs integrated services and automation on Chameleon | `k8s/ml-system/base/`, `k8s/ml-system/overlays/staging/`, `k8s/ml-system/overlays/canary/`, `k8s/ml-system/overlays/production/` |
| Test orchestration | Final project test runner | local script | Generates data, starts services, sends traffic, posts quality metrics, tests rollout scripts, and opens service UIs | `scripts/final_project_test.sh`, `scripts/generate_actual_bank_import.py`, `scripts/simulate_promotion_rollback.py` |

## Local Service Startup

Recommended local end-to-end command from the repository root:

```bash
bash scripts/final_project_test.sh
```

The script runs the local test stack in this order:

```text
Generate bank import and serving batch data
        |
Start serving, Prometheus, and Grafana through Docker Compose
        |
Send realistic /predict_batch traffic to serving
        |
Post data quality metrics to /monitor/data-quality
        |
Run promotion and rollback simulation
        |
Start Actual dev server if needed
        |
Open Actual, serving docs, serving summary, Prometheus, and Grafana overview
```

Local service URLs:

| URL | Service |
| --- | --- |
| `http://127.0.0.1:3001` | Actual frontend |
| `http://127.0.0.1:5006` | Actual server-dev/sync server |
| `http://127.0.0.1:8000/docs` | Serving API docs |
| `http://127.0.0.1:8000/monitor/summary` | Serving behavior summary |
| `http://127.0.0.1:8000/monitor/decision` | Promotion/rollback decision |
| `http://127.0.0.1:9090` | Prometheus |
| `http://127.0.0.1:3000/d/actual-ml-system-overview/actual-ml-system-overview` | Grafana system overview |

## Actual Integration Services

### Actual frontend

Purpose:

- lets users import bank transactions into an account
- requests predictions for visible account transactions
- displays `AI Suggestion` and `All Top-3`
- lets users click a suggested category
- updates the transaction category after a click

Main files:

```text
packages/desktop-client/src/components/accounts/Account.tsx
packages/desktop-client/src/components/transactions/TransactionsTable.tsx
packages/desktop-client/src/components/Titlebar.tsx
```

Important behavior:

- `Account.tsx` builds prediction payloads from transaction fields and calls Actual core methods.
- `TransactionsTable.tsx` renders the prediction columns and user-selectable Top-3 buttons.
- `Titlebar.tsx` shows the category review count in a non-blocking way.

### Actual core bridge

Purpose:

- exposes frontend-callable ML methods
- calls serving `/predict` and `/predict_batch`
- stores predictions in local SQLite tables
- records feedback when a user chooses a category
- mirrors feedback to serving for monitoring

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

## Serving API

Purpose:

- owns online model inference
- owns feedback ingestion for monitoring
- owns model behavior summaries and rollout decisions
- exposes Prometheus metrics
- reloads model after promotion or rollback

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

Model configuration is controlled by environment variables in:

```text
serving/.env.example
serving/app/config.py
serving/docker-compose.yml
k8s/ml-system/base/kustomization.yaml
```

Key variables:

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

## Model Artifacts

Purpose:

- provide the model files loaded by serving
- keep the source and optimized model variants together
- document class labels and artifact metadata

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

## Monitoring Services

### Prometheus

Purpose:

- scrapes serving `/metrics`
- stores time-series metrics
- evaluates rollout and data quality alerts
- defines summary recording rules used by Grafana

Main files:

```text
serving/monitoring/prometheus.yml
serving/monitoring/prometheus-alerts.yml
serving/docker-compose.yml
```

Important scraped/exported metrics:

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

Summary recording rules:

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

### Grafana

Purpose:

- provides dashboards for service health, model outputs, user feedback, data quality, and rollout alerts
- gives one overview page for demos and production checks

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

Main dashboard:

```text
http://127.0.0.1:3000/d/actual-ml-system-overview/actual-ml-system-overview
```

The overview dashboard summarizes:

- serving health
- request rate
- p50 and p95 latency
- 5xx error rate
- feedback volume
- data quality pass/fail status
- Top-1 and Top-3 acceptance
- predicted category distribution
- average confidence and low-confidence ratio
- firing rollout alerts
- requests by endpoint and status
- latest data quality metrics

## Data Quality Services

Purpose:

- evaluate data quality at ingestion
- evaluate training-set quality before retraining
- monitor online inference drift
- publish data quality status into serving and Prometheus/Grafana

Main files:

```text
data/data_quality_check.py
data/ingest.py
data/batch_pipeline.py
data/online_features.py
data/feedback_collector.py
k8s/ml-system/base/data-quality-cronjob.yaml
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

Checks published to serving:

```text
ingestion
training_set
online_drift
```

## Training And Model Promotion Services

Purpose:

- collect/compile training data
- run data quality gates
- train a challenger model
- evaluate task-specific metrics
- promote challenger if it passes thresholds and beats current deployed model
- archive rollback metadata

Main files:

```text
training/build_training_set.py
training/train_model.py
scripts/run_mlops_pipeline.py
scripts/promote_model.py
scripts/rollback_model.py
scripts/simulate_promotion_rollback.py
.github/workflows/mlops-automation.yml
docker/mlops.Dockerfile
k8s/ml-system/base/training-pipeline-cronjob.yaml
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

Model deployment directories used by scripts:

```text
artifacts/challenger
artifacts/deployed
artifacts/archive
```

## Rollout And Rollback Services

Purpose:

- convert serving monitoring results into concrete promotion/rollback actions
- support dry-run mode before mutating model artifacts
- reload serving model after a successful action

Main files:

```text
serving/tools/execute_rollout_action.py
scripts/promote_model.py
scripts/rollback_model.py
serving/app/main.py
serving/app/runtime.py
k8s/ml-system/base/rollout-decision-cronjob.yaml
```

Decision source:

```text
GET /monitor/decision
```

Possible actions:

| Action | Meaning |
| --- | --- |
| `hold` | No action required |
| `promote_candidate` | Candidate environment has enough traffic, feedback, latency, error-rate, and acceptance evidence for promotion |
| `rollback_active` | Production behavior crossed rollback thresholds |

Dry-run:

```bash
python3 serving/tools/execute_rollout_action.py \
  --monitor-url http://127.0.0.1:8000/monitor/decision
```

Execute action:

```bash
python3 serving/tools/execute_rollout_action.py \
  --execute \
  --monitor-url http://127.0.0.1:8000/monitor/decision \
  --reload-url http://127.0.0.1:8000/admin/reload-model
```

Kubernetes schedule:

```text
CronJob: serving-rollout-decision
Schedule: */15 * * * *
File: k8s/ml-system/base/rollout-decision-cronjob.yaml
```

Rollback triggers include:

```text
p95 latency above threshold
5xx error rate above threshold
Top-1 acceptance below threshold
Top-3 acceptance below threshold
```

## Kubernetes Services

Purpose:

- run the integrated system on Chameleon
- keep environments separated
- support staging, canary, and production
- automate data quality, retraining/promotion, and rollout decisions

Base files:

```text
k8s/ml-system/base/storage.yaml
k8s/ml-system/base/actual-sync.yaml
k8s/ml-system/base/serving.yaml
k8s/ml-system/base/monitoring.yaml
k8s/ml-system/base/data-quality-cronjob.yaml
k8s/ml-system/base/training-pipeline-cronjob.yaml
k8s/ml-system/base/rollout-decision-cronjob.yaml
k8s/ml-system/base/kustomization.yaml
```

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

Environment behavior:

| Environment | Namespace | Serving context | Purpose |
| --- | --- | --- | --- |
| staging | `actual-ml-staging` | `ROLLOUT_CONTEXT=staging` | Validate service wiring with lower thresholds |
| canary | `actual-ml-canary` | `ROLLOUT_CONTEXT=candidate` | Evaluate candidate model for promotion |
| production | `actual-ml-production` | `ROLLOUT_CONTEXT=production` | Main deployed model, rollback if behavior degrades |

Kubernetes services:

| Service | File | Port |
| --- | --- | --- |
| `actual-sync` | `k8s/ml-system/base/actual-sync.yaml` | `5006` |
| `smartcat-serving` | `k8s/ml-system/base/serving.yaml` | `8000` |
| `prometheus` | `k8s/ml-system/base/monitoring.yaml` | `9090` |
| `grafana` | `k8s/ml-system/base/monitoring.yaml` | `3000` |

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

## Docker Compose Services

Purpose:

- run local serving and monitoring stack without Kubernetes
- mount dashboard and Prometheus configs directly from the repository

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

Start via orchestration helper:

```bash
cd serving
python3 run.py monitor-up
```

## Test Data And Traffic Services

Purpose:

- generate realistic user-like transactions
- test Actual import flow
- test serving batch prediction traffic
- provide synthetic training data for local pipeline tests

Main files:

```text
scripts/generate_actual_bank_import.py
scripts/final_project_test.sh
serving/runtime/batch_input_realistic.json
artifacts/test-data/actual_bank_transactions.qif
artifacts/test-data/synthetic_training_transactions.csv
```

Generated artifacts are runtime/test outputs and are not required in source
control.

## Requirement Coverage

| Project requirement | System component | Evidence files |
| --- | --- | --- |
| Integrated system inside chosen open source service | Actual account transaction table calls serving and displays Top-3 predictions | `Account.tsx`, `TransactionsTable.tsx`, `packages/loot-core/src/server/ml/` |
| End-to-end production data to inference | Bank import -> Actual transactions -> `/predict_batch` -> UI columns | `scripts/generate_actual_bank_import.py`, `Account.tsx`, `serving/app/main.py` |
| Feedback capture | User category click records feedback locally and mirrors to serving | `TransactionsTable.tsx`, `packages/loot-core/src/server/ml/store.ts`, `serving/app/main.py` |
| Retraining and evaluation | Pipeline builds splits, checks quality, trains, evaluates, and promotes | `scripts/run_mlops_pipeline.py`, `training/train_model.py`, `scripts/promote_model.py` |
| Monitoring model output | Prediction confidence and class distribution metrics | `serving/app/main.py`, `serving/monitoring/grafana/dashboards/prediction_monitoring.json` |
| Monitoring operational metrics | Request rate, latency, errors, health | `serving/monitoring/prometheus-alerts.yml`, `serving/monitoring/grafana/dashboards/service_monitoring.json` |
| Monitoring user feedback | Top-1/Top-3 acceptance and feedback volume | `serving/app/main.py`, `serving/monitoring/grafana/dashboards/feedback_monitoring.json` |
| Data quality monitoring | Ingestion, training-set, and online drift checks | `data/data_quality_check.py`, `k8s/ml-system/base/data-quality-cronjob.yaml` |
| Promotion trigger | Serving decision recommends and executes promotion | `serving/tools/execute_rollout_action.py`, `scripts/promote_model.py` |
| Rollback trigger | Serving decision recommends and executes rollback | `serving/tools/execute_rollout_action.py`, `scripts/rollback_model.py` |
| Automation | CI workflow and K8s CronJobs | `.github/workflows/mlops-automation.yml`, `k8s/ml-system/base/*.yaml` |
| Kubernetes deployment | staging/canary/production overlays | `k8s/ml-system/overlays/` |
| Safeguarding plan | fairness, transparency, privacy, robustness, accountability mechanisms | `docs/ml-system/SAFEGUARDING.md` |

## Operational Checklist

Before a demo or Chameleon run:

1. Confirm serving is healthy:

```bash
curl -s http://127.0.0.1:8000/readyz
```

2. Confirm metrics are exposed:

```bash
curl -s http://127.0.0.1:8000/metrics | head
```

3. Confirm Prometheus config/rules are valid:

```bash
docker run --rm --entrypoint promtool \
  -v "$PWD/serving/monitoring:/etc/prometheus:ro" \
  prom/prometheus:latest check config /etc/prometheus/prometheus.yml
```

4. Confirm dashboards are valid JSON:

```bash
python3 -m json.tool serving/monitoring/grafana/dashboards/system_overview.json >/tmp/system_overview.json
```

5. Confirm K8s manifests render:

```bash
kubectl kustomize k8s/ml-system/overlays/staging
kubectl kustomize k8s/ml-system/overlays/canary
kubectl kustomize k8s/ml-system/overlays/production
```
