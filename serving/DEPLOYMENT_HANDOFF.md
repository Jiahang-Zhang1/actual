# SmartCat Serving Handoff

## What this container does

- Loads one approved model artifact at a time (`baseline`, `onnx`, or `onnx_dynamic_quant`)
- Exposes stable HTTP endpoints:
  - `POST /predict`
  - `POST /predict_batch`
  - `POST /feedback`
  - `GET /healthz`
  - `GET /readyz`
  - `GET /versionz`
  - `GET /monitor/summary`
  - `GET /monitor/decision`
  - `GET /metrics`

## Input contract for data team

Use `POST /predict_batch` with:

```json
{
  "items": [
    {
      "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
      "transaction_description_clean": "starbucks store 1458 new york ny",
      "country": "US",
      "currency": "USD",
      "description_length": 6
    }
  ]
}
```

The current model path consumes `transaction_description`, `transaction_description_clean` (fallback),
`merchant_text` (fallback), `country`, and `currency`. The extra metadata fields are accepted and ignored
unless the training/feature contract is expanded later.

## External URLs

### Docker Compose on Chameleon VM

- Serving docs: `http://<floating-ip>:8000/docs`
- Prometheus: `http://<floating-ip>:9090`
- Grafana: `http://<floating-ip>:3000`

### Kubernetes (NodePort)

Apply `k8s/serving.yaml`, then use:

- Serving docs: `http://<node-ip>:30080/docs`

## Environment variables DevOps needs

- `BACKEND_KIND`
- `MODEL_PATH`
- `SOURCE_MODEL_PATH`
- `MODEL_VERSION`
- `CODE_VERSION`
- `WEB_CONCURRENCY`
- `SERVICE_PORT`
- `ROLLOUT_CONTEXT`
- promotion / rollback threshold vars from `.env.example`

## Rollout decision model

- `ROLLOUT_CONTEXT=candidate`: `/monitor/decision` recommends `promote_candidate` only if latency,
  error rate, and feedback thresholds are met.
- `ROLLOUT_CONTEXT=production`: `/monitor/decision` recommends `rollback_active` if latency,
  error rate, or acceptance thresholds are violated.
