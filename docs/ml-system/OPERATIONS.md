# Smart Transaction Categorization Operations

## Local Final Demo

Fresh Chameleon VM:

```bash
bash bootstrap_chameleon.sh
```

The bootstrap script installs Docker/Compose, Node 22/Yarn, Python runtime
dependencies, starts the local service stack, generates test traffic, and prints
the service URLs.

Existing development machine:

```bash
bash scripts/final_project_test.sh
```

This generates realistic bank import data, starts serving/Prometheus/Grafana
when Docker Compose is available, sends realistic `/predict_batch` traffic,
publishes a data quality result, exercises promotion/rollback scripts, and
opens the main service pages.

Generated import file:

```text
artifacts/test-data/actual_bank_transactions.qif
```

Use Actual's built-in `Create test file`, open an account, then import the QIF
as bank transactions. This avoids relying on the demo budget while still
testing the normal user flow.

For the full team walkthrough, service inventory, requirement mapping, and
testing sequence, see `serving/README.md`.

## Automated Pipeline

Local:

```bash
python scripts/run_mlops_pipeline.py --synthetic-bootstrap
```

CI:

```text
.github/workflows/mlops-automation.yml
```

Kubernetes:

```text
k8s/ml-system/base/training-pipeline-cronjob.yaml
```

## Rollout Trigger

Dry-run:

```bash
python serving/tools/execute_rollout_action.py \
  --monitor-url http://localhost:8000/monitor/decision
```

Execute:

```bash
python serving/tools/execute_rollout_action.py --execute \
  --monitor-url http://localhost:8000/monitor/decision
```

Simulation without training dependencies:

```bash
python scripts/simulate_promotion_rollback.py
```
