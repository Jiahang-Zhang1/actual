# Smart Transaction Categorization Update Report

Date: 2026-04-23

## Overview

This update focused on turning the current Smart Transaction Categorization system
into a more robust end-to-end flow for manual demo usage, sparse user input, and
repeatable training evaluation.

The changes were intentionally split into two layers:

1. online robustness and demo safety for Actual + SmartCat serving
2. stronger training evaluation and gating without breaking the existing serving contract

## What Changed

### 1. Sparse Manual-Entry Prediction Support

The online system no longer depends on a fully populated transaction description.
Prediction now works when the user only provides partial information such as:

- `notes`
- `imported_description`
- `account_id`
- `amount`
- `currency`

Fallback description construction was added across:

- `serving/app/feature_adapter.py`
- `serving/app/main.py`
- `packages/loot-core/src/server/ml/service.ts`
- `packages/loot-core/src/server/transactions/app.ts`
- `packages/desktop-client/src/components/accounts/Account.tsx`
- `data/online_features.py`

This means TA/professor manual interaction in Actual is now more resilient and
the API docs page is also usable for sparse request testing.

### 2. Confidence and Top-3 Consistency

The system now normalizes prediction display behavior so that:

- `predicted_category_id` is aligned with the highest-scoring Top-3 candidate
- `confidence` uses the Top-1 score consistently
- frontend display and serving response agree on the same confidence value

This prevents misleading UI behavior where `confidence` and `top_categories[0]`
previously diverged.

### 3. Synthetic Traffic Isolation

Benchmark and smoke traffic can now be labeled and excluded from online
monitoring and rollout windows using `X-Actual-Traffic-Source`.

This was added so that:

- compressed load tests do not pollute `monitor/summary`
- feedback acceptance metrics remain meaningful
- Grafana and Prometheus reflect live-decision traffic instead of mixed traffic

Relevant updates were made in:

- `serving/app/main.py`
- `serving/monitoring/prometheus-alerts.yml`
- `serving/tools/benchmark_http.py`
- `serving/tools/benchmark_arrivals.py`
- `scripts/run_week_simulation.py`

### 4. Training Pipeline Strengthening

The training path now better matches the deployed serving behavior.

Changes include:

- sparse/manual-entry feedback rows are no longer dropped during dataset build
- training feature text uses the same fallback and description-source logic as serving
- new evaluation script emits richer artifacts
- new gate script applies challenger quality thresholds before promotion

New files:

- `training/evaluate_model.py`
- `training/gate_model.py`

Updated pipeline:

- `scripts/run_mlops_pipeline.py`

The pipeline now runs:

1. dataset quality checks
2. training
3. evaluation
4. gating
5. model artifact export
6. promotion decision

### 5. Test Coverage Added

New and updated tests now cover:

- sparse manual-entry request handling
- fallback description generation
- confidence normalization
- synthetic traffic exclusion from monitor summaries

## Validation Performed

The following validations were run locally:

- `pytest serving/tests/test_feature_adapter.py serving/tests/test_api_contract.py`
- `yarn workspace @actual-app/core run test:node src/server/ml/service.test.ts`
- `yarn typecheck`
- `python3 scripts/run_mlops_pipeline.py --workspace-dir artifacts/mlops-pipeline-integration-smoke --source-training-csv artifacts/test-data/synthetic_training_transactions.csv --deployed-dir artifacts/mlops-pipeline-integration-smoke/deployed --archive-dir artifacts/mlops-pipeline-integration-smoke/archive`

Results:

- serving tests: passed
- core ML service tests: passed
- typecheck: passed
- local pipeline smoke: passed

The local pipeline smoke also produced:

- `metrics.json`
- `per_class_metrics.json`
- `slice_metrics.json`
- `gate_result.json`
- `validation_predictions.csv`
- `test_predictions.csv`

## Deployment Notes

The current Chameleon deployment already includes the sparse/manual-entry serving
hardening and the synthetic-traffic monitoring separation.

The new training evaluation and gate integration is now in the repository and
ready for repeatable retrain/evaluate flows, but it does not by itself claim
that a new production model has already been retrained and promoted on Chameleon.

## Remaining Gaps

The system is materially stronger than before, but these areas still deserve
follow-up work:

- curated seed dataset expansion for real manual-entry behavior
- label taxonomy alignment if external training repos are imported
- confidence calibration beyond raw model score usage
- MLflow registry population and alias evidence capture on Chameleon
- more realistic long-window simulations using non-synthetic user data

## Bottom Line

This update moved the project from a brittle demo path toward a more reliable
course-project system:

- users can interact with the product more naturally
- sparse input no longer breaks inference
- monitoring is cleaner
- training evaluation is more defensible
- future retraining and promotion decisions have better evidence
