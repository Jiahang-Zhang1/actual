#!/usr/bin/env bash
set -euo pipefail

# Create the Chameleon security group/rule requested by DevOps and attach it to
# node1's Neutron port. Run this from an environment with the OpenStack RC file
# sourced, or perform the equivalent steps in Horizon.

SECURITY_GROUP="${SECURITY_GROUP:-allow-30443-proj08}"
PORT_ID="${PORT_ID:-4e02dcef-5540-4b5d-aed4-3ca53e53c95b}"
PORT="${PORT:-30443}"

if ! command -v openstack >/dev/null 2>&1; then
  echo "openstack CLI not found. Source your Chameleon OpenStack RC and install python-openstackclient, or use Horizon." >&2
  exit 1
fi

if ! openstack security group show "${SECURITY_GROUP}" >/dev/null 2>&1; then
  openstack security group create "${SECURITY_GROUP}" \
    --description "Allow HTTPS NodePort ${PORT} for proj08 Actual Budget"
fi

# Duplicate rules return a non-zero error on some OpenStack versions; that is
# safe because the desired rule already exists.
openstack security group rule create "${SECURITY_GROUP}" \
  --ingress \
  --ethertype IPv4 \
  --protocol tcp \
  --dst-port "${PORT}:${PORT}" \
  --remote-ip 0.0.0.0/0 || true

new_group_id="$(openstack security group show "${SECURITY_GROUP}" -f value -c id)"
existing_group_ids="$(
  openstack port show "${PORT_ID}" -f json -c security_group_ids | python3 -c '
import json, sys
data = json.load(sys.stdin).get("security_group_ids", [])
if isinstance(data, str):
    data = [part.strip() for part in data.replace(",", " ").split() if part.strip()]
print("\n".join(data))
'
)"

port_set_args=()
seen=" ${new_group_id} "
port_set_args+=(--security-group "${new_group_id}")
while IFS= read -r group_id; do
  [[ -z "${group_id}" ]] && continue
  if [[ "${seen}" != *" ${group_id} "* ]]; then
    port_set_args+=(--security-group "${group_id}")
    seen+=" ${group_id} "
  fi
done <<< "${existing_group_ids}"

openstack port set "${port_set_args[@]}" "${PORT_ID}"

echo "Security group ${SECURITY_GROUP} allows TCP ${PORT} and is attached to port ${PORT_ID}."
