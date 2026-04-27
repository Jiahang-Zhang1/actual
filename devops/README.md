# DevOps K8s HTTPS entrypoint

This folder contains the K8s manifests used to expose the modified Actual Budget UI and the SmartCat serving API on Chameleon.

## What this solves

Chrome only exposes `SharedArrayBuffer` when Actual is loaded from a secure context and the response has cross-origin isolation headers. A public HTTP NodePort such as `http://129.114.26.190:30083` is not enough, even when the app itself is healthy.

The nginx proxy in this folder terminates HTTPS on NodePort `30443`, forwards traffic to `actual-budget.actual-budget.svc.cluster.local:8083`, and sends exactly one copy of each required header:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Files

| File                            | Purpose                                                                                                               |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `actual-budget-deployment.yaml` | Runs the Actual Budget image on container port `5006` and exposes a private ClusterIP service on service port `8083`. |
| `nginx-https-proxy.yaml`        | Runs nginx on port `443`, mounts a TLS Secret, forwards to Actual service port `8083`, and exposes NodePort `30443`.  |
| `setup-nginx-https.sh`          | Generates a self-signed certificate, stores it in K8s Secret `actual-nginx-tls`, then applies the manifests.          |
| `chameleon-open-30443.sh`       | Optional OpenStack CLI helper to create/attach the `allow-30443-proj08` security group.                               |
| `smartcat-serving.yaml`         | Runs the SmartCat model serving service on port `8000` / NodePort `30090`.                                            |
| `smartcat-hpa.yaml`             | Adds HPA rules for SmartCat serving.                                                                                  |
| `alert-rules.yaml`              | Adds Prometheus alert rules if the monitoring CRDs are installed.                                                     |
| `kustomization.yaml`            | Lets the team deploy the manifests as one bundle.                                                                     |

## Deploy on node1

Run this from node1 where `/etc/kubernetes/admin.conf` exists:

```bash
git pull origin serving
bash devops/setup-nginx-https.sh
```

If running from another machine, set `KUBECONFIG` first:

```bash
export KUBECONFIG=/path/to/admin.conf
bash devops/setup-nginx-https.sh
```

## Open Chameleon security group

If the OpenStack CLI is configured:

```bash
bash devops/chameleon-open-30443.sh
```

Equivalent Horizon action:

1. Create security group `allow-30443-proj08`.
2. Add ingress TCP rule `30443` from `0.0.0.0/0`.
3. Attach that security group to node1 port `4e02dcef-5540-4b5d-aed4-3ca53e53c95b`.

## Test URLs

Because this uses a self-signed certificate, the browser will show a certificate warning. Accept it for the demo.

```text
https://129.114.26.190:30443
https://129.114.26.190.sslip.io:30443
https://actual.129.114.26.190.sslip.io:30443
```

The old HTTP URL should not be used for the Actual UI:

```text
http://129.114.26.190:30083
```
