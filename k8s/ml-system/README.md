# Integrated K8s Deployment

This directory converges the local Docker Compose setup into one Kubernetes
deployment for the Actual smart transaction categorization system.

## Environments

Use one overlay per environment:

```bash
kubectl apply -k k8s/ml-system/overlays/staging
kubectl apply -k k8s/ml-system/overlays/canary
kubectl apply -k k8s/ml-system/overlays/production
```

The overlays create separate namespaces:

- `actual-ml-staging`
- `actual-ml-canary`
- `actual-ml-production`

Each environment includes:

- Actual sync service with `ACTUAL_ML_SERVICE_URL` pointed at serving
- FastAPI serving deployment with `/predict`, `/predict_batch`, `/feedback`, `/monitor/*`, and `/metrics`
- Prometheus and Grafana
- data quality CronJob for ingestion, training-set, and online-drift checks
- retrain/evaluate/promote CronJob
- rollout decision CronJob owned by serving

## Chameleon Notes

For an `m1.large`, keep production at two serving replicas unless the node has
extra headroom. Prometheus and Grafana can run in the same namespace for the
class demo, but a larger Chameleon deployment should move monitoring into a
shared namespace.

Recommended port-forward commands:

```bash
kubectl -n actual-ml-production port-forward svc/actual-sync 5006:5006
kubectl -n actual-ml-production port-forward svc/smartcat-serving 8000:8000
kubectl -n actual-ml-production port-forward svc/prometheus 9090:9090
kubectl -n actual-ml-production port-forward svc/grafana 3000:3000
```

## Promotion And Rollback

The training CronJob writes challenger artifacts into the shared `ml-artifacts`
PVC, evaluates quality gates, then calls `scripts/promote_model.py`.

The serving rollout CronJob calls:

```bash
python serving/tools/execute_rollout_action.py --execute \
  --monitor-url http://smartcat-serving:8000/monitor/decision
```

Serving recommends promotion only in candidate context when enough traffic,
feedback, latency, error-rate, and acceptance thresholds pass. Production
recommends rollback when latency, error rate, or feedback acceptance crosses
rollback thresholds.
