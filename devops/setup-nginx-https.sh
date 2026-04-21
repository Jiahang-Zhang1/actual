#!/usr/bin/env bash
set -euo pipefail

# Generate a self-signed TLS certificate, store it in the Kubernetes Secret that
# the nginx reverse proxy mounts, then apply the devops manifests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAMESPACE="${NAMESPACE:-actual-budget}"
SECRET_NAME="${SECRET_NAME:-actual-nginx-tls}"
FLOATING_IP="${FLOATING_IP:-129.114.26.190}"
PRIMARY_DNS="${PRIMARY_DNS:-${FLOATING_IP}.sslip.io}"
ACTUAL_DNS="${ACTUAL_DNS:-actual.${FLOATING_IP}.sslip.io}"
KUBECONFIG_PATH="${KUBECONFIG:-}"

if [[ -z "${KUBECONFIG_PATH}" && -f /etc/kubernetes/admin.conf ]]; then
  KUBECONFIG_PATH=/etc/kubernetes/admin.conf
fi

kubectl_args=()
if [[ -n "${KUBECONFIG_PATH}" ]]; then
  kubectl_args+=(--kubeconfig "${KUBECONFIG_PATH}")
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

cat > "${tmpdir}/openssl.cnf" <<CONF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[dn]
CN = ${PRIMARY_DNS}

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${PRIMARY_DNS}
DNS.2 = ${ACTUAL_DNS}
IP.1 = ${FLOATING_IP}
CONF

openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
  -keyout "${tmpdir}/tls.key" \
  -out "${tmpdir}/tls.crt" \
  -config "${tmpdir}/openssl.cnf"

kubectl "${kubectl_args[@]}" create namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl "${kubectl_args[@]}" apply -f -

kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" create secret tls "${SECRET_NAME}" \
  --cert="${tmpdir}/tls.crt" \
  --key="${tmpdir}/tls.key" \
  --dry-run=client -o yaml | kubectl "${kubectl_args[@]}" apply -f -

kubectl "${kubectl_args[@]}" apply -f "${SCRIPT_DIR}/actual-budget-deployment.yaml"
kubectl "${kubectl_args[@]}" apply -f "${SCRIPT_DIR}/smartcat-serving.yaml"
kubectl "${kubectl_args[@]}" apply -f "${SCRIPT_DIR}/smartcat-hpa.yaml"
kubectl "${kubectl_args[@]}" apply -f "${SCRIPT_DIR}/nginx-https-proxy.yaml"

if kubectl "${kubectl_args[@]}" api-resources --api-group=monitoring.coreos.com \
  | grep -q '^prometheusrules'; then
  kubectl "${kubectl_args[@]}" apply -f "${SCRIPT_DIR}/alert-rules.yaml"
else
  echo "PrometheusRule CRD not found; skipping devops/alert-rules.yaml."
fi

cat <<MSG
nginx HTTPS proxy deployed.
Open one of these URLs and accept the self-signed certificate warning:
  https://${FLOATING_IP}:30443
  https://${PRIMARY_DNS}:30443
  https://${ACTUAL_DNS}:30443
MSG
