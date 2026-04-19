# SmartCat Serving K8s handoff

Apply on a K3s/Kubernetes cluster:

```bash
kubectl apply -f k8s/serving.yaml
```

External endpoint (NodePort):

```text
http://<node-ip>:30080/predict
http://<node-ip>:30080/predict_batch
```

Prometheus/Grafana are kept in Docker Compose for local/VM benchmarking in this repo.
DevOps can either keep them separate or port the same images/config to Kubernetes.
