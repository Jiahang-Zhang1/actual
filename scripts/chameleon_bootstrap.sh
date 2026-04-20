#!/usr/bin/env bash
set -euo pipefail

# Chameleon VM bootstrap for the integrated Actual smart categorization demo.
# The script is intentionally idempotent so teammates can re-run it after pulls.
ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

RUN_ACTUAL=1
RUN_TRAFFIC=1
INSTALL_YARN_DEPS=1
INSTALL_PYTHON_DEPS=1
INSTALL_DOCKER=1
TEST_ROWS="${TEST_ROWS:-1500}"
PUBLIC_HOST="${PUBLIC_HOST:-}"
SERVING_URL="${SERVING_URL:-http://127.0.0.1:8000}"

usage() {
  cat <<'USAGE'
Usage: bash bootstrap_chameleon.sh [options]

Options:
  --serving-only          Install dependencies and start serving/monitoring only.
  --skip-actual           Do not install Yarn dependencies or start Actual web/server-dev.
  --skip-traffic          Do not generate demo traffic/data quality/promotion simulation.
  --skip-yarn-install     Skip yarn install even when Actual is enabled.
  --skip-python-deps      Skip the local Python virtualenv dependency install.
  --skip-docker-install   Require Docker to already be installed.
  --test-rows N           Number of synthetic bank rows to generate. Default: 1500.
  --public-host HOST      Hostname or floating IP printed in the final URL summary.
  -h, --help              Show this help text.

Common first run:
  git clone <repo-url>
  cd actual
  git switch serving
  bash bootstrap_chameleon.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serving-only)
      RUN_ACTUAL=0
      RUN_TRAFFIC=0
      INSTALL_YARN_DEPS=0
      shift
      ;;
    --skip-actual)
      RUN_ACTUAL=0
      INSTALL_YARN_DEPS=0
      shift
      ;;
    --skip-traffic)
      RUN_TRAFFIC=0
      shift
      ;;
    --skip-yarn-install)
      INSTALL_YARN_DEPS=0
      shift
      ;;
    --skip-python-deps)
      INSTALL_PYTHON_DEPS=0
      shift
      ;;
    --skip-docker-install)
      INSTALL_DOCKER=0
      shift
      ;;
    --test-rows)
      TEST_ROWS="${2:?--test-rows requires a value}"
      shift 2
      ;;
    --public-host)
      PUBLIC_HOST="${2:?--public-host requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf '\nWARNING: %s\n' "$*" >&2
}

sudo_run() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

have_command() {
  command -v "$1" >/dev/null 2>&1
}

apt_install() {
  sudo_run env DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
}

ensure_base_packages() {
  if ! have_command apt-get; then
    warn "This bootstrap currently targets Ubuntu/Debian Chameleon images with apt-get."
    return
  fi

  log "Installing base system packages"
  sudo_run apt-get update
  apt_install ca-certificates curl gnupg lsb-release git jq unzip build-essential python3 python3-pip python3-venv
}

install_docker_from_distro() {
  if apt-cache show docker-compose-v2 >/dev/null 2>&1; then
    apt_install docker.io docker-compose-v2
  elif apt-cache show docker-compose-plugin >/dev/null 2>&1; then
    apt_install docker.io docker-compose-plugin
  else
    return 1
  fi
}

install_docker_from_official_repo() {
  # Fallback for Ubuntu images whose default apt repo does not include Compose v2.
  local codename
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")"
  sudo_run install -m 0755 -d /etc/apt/keyrings
  sudo_run rm -f /etc/apt/keyrings/docker.gpg
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo_run gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  sudo_run chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${codename} stable" \
    | sudo_run tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo_run apt-get update
  apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

ensure_docker() {
  if have_command docker && docker compose version >/dev/null 2>&1; then
    log "Docker and Docker Compose are already installed"
  elif [[ "$INSTALL_DOCKER" -eq 1 ]]; then
    log "Installing Docker and Docker Compose"
    install_docker_from_distro || install_docker_from_official_repo
  else
    echo "Docker Compose is required but --skip-docker-install was provided." >&2
    exit 1
  fi

  if have_command systemctl; then
    sudo_run systemctl enable --now docker || true
  else
    sudo_run service docker start || true
  fi

  local bootstrap_user
  bootstrap_user="${SUDO_USER:-${USER}}"
  if ! docker ps >/dev/null 2>&1; then
    log "Adding ${bootstrap_user} to docker group"
    sudo_run usermod -aG docker "$bootstrap_user" || true

    if [[ -z "${CHAMELEON_BOOTSTRAP_REEXEC:-}" ]] && have_command sg && [[ "${EUID}" -ne 0 ]]; then
      # Re-enter the script with docker group permissions without requiring logout/login.
      local quoted_args=""
      if [[ "${#ORIGINAL_ARGS[@]}" -gt 0 ]]; then
        printf -v quoted_args ' %q' "${ORIGINAL_ARGS[@]}"
      fi
      log "Re-running bootstrap inside the docker group"
      exec sg docker -c "cd $(printf '%q' "$ROOT_DIR") && CHAMELEON_BOOTSTRAP_REEXEC=1 bash scripts/chameleon_bootstrap.sh${quoted_args}"
    fi
  fi

  docker ps >/dev/null
  docker compose version >/dev/null
}

ensure_node_and_yarn() {
  if [[ "$RUN_ACTUAL" -eq 0 ]]; then
    return
  fi

  local node_major="0"
  if have_command node; then
    node_major="$(node -p 'Number(process.versions.node.split(".")[0])' 2>/dev/null || echo 0)"
  fi

  if [[ "$node_major" -lt 22 ]]; then
    log "Installing Node.js 22"
    if [[ "${EUID}" -eq 0 ]]; then
      curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    else
      curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    fi
    apt_install nodejs
  else
    log "Node.js $(node --version) is already compatible"
  fi

  if ! have_command corepack; then
    log "Installing corepack"
    sudo_run npm install -g corepack
  fi

  log "Enabling Yarn through corepack"
  sudo_run corepack enable || corepack enable
  corepack prepare yarn@4.13.0 --activate
}

ensure_python_env() {
  if [[ "$INSTALL_PYTHON_DEPS" -eq 0 ]]; then
    return
  fi

  log "Preparing local Python virtualenv"
  python3 -m venv .venv-chameleon
  # shellcheck disable=SC1091
  source .venv-chameleon/bin/activate
  python -m pip install --upgrade pip wheel
  python -m pip install -r serving/requirements.txt
}

install_yarn_dependencies() {
  if [[ "$RUN_ACTUAL" -eq 0 || "$INSTALL_YARN_DEPS" -eq 0 ]]; then
    return
  fi

  log "Installing Yarn workspace dependencies"
  YARN_ENABLE_IMMUTABLE_INSTALLS=false yarn install
}

configure_serving_env() {
  log "Preparing serving environment"
  mkdir -p serving/runtime artifacts/test-data artifacts/test-rollout artifacts/chameleon
  if [[ ! -f serving/.env ]]; then
    cp serving/.env.example serving/.env
  fi
}

wait_for_url() {
  local url="$1"
  local label="$2"
  local retries="${3:-90}"
  for _ in $(seq 1 "$retries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "${label} is ready: ${url}"
      return 0
    fi
    sleep 2
  done
  echo "${label} did not become ready: ${url}" >&2
  return 1
}

start_serving_stack() {
  log "Starting serving, MLflow, Prometheus, and Grafana"
  (cd serving && python3 run.py monitor-up)
  wait_for_url "${SERVING_URL%/}/readyz" "serving" 90
  wait_for_url "http://127.0.0.1:5000" "MLflow" 90 || true
  wait_for_url "http://127.0.0.1:9090/-/ready" "Prometheus" 60 || true
  wait_for_url "http://127.0.0.1:3000/api/health" "Grafana" 60 || true
}

run_demo_traffic() {
  if [[ "$RUN_TRAFFIC" -eq 0 ]]; then
    return
  fi

  log "Generating realistic bank import and serving batch data"
  python3 scripts/generate_actual_bank_import.py \
    --rows "$TEST_ROWS" \
    --csv-output artifacts/test-data/actual_bank_transactions.csv \
    --qif-output artifacts/test-data/actual_bank_transactions.qif \
    --batch-output serving/runtime/batch_input_realistic.json \
    --training-output artifacts/test-data/synthetic_training_transactions.csv

  log "Sending realistic predict_batch traffic"
  curl -fsS \
    -H "Content-Type: application/json" \
    -d @serving/runtime/batch_input_realistic.json \
    "${SERVING_URL%/}/predict_batch" >/tmp/actual_predict_batch_response.json

  log "Publishing data quality metrics"
  python3 data/data_quality_check.py ingestion \
    --input artifacts/test-data/synthetic_training_transactions.csv \
    --output-json artifacts/test-data/ingestion_quality.json \
    --post-url "${SERVING_URL%/}/monitor/data-quality" || true

  log "Exercising promotion and rollback simulation"
  python3 scripts/simulate_promotion_rollback.py
}

start_actual_dev_server() {
  if [[ "$RUN_ACTUAL" -eq 0 ]]; then
    return
  fi

  if curl -fsS http://127.0.0.1:3001 >/dev/null 2>&1; then
    log "Actual frontend is already running"
    return
  fi

  log "Starting Actual web frontend and server-dev in the background"
  mkdir -p artifacts/chameleon
  ACTUAL_ML_SERVICE_URL="${SERVING_URL%/}" \
  ACTUAL_HOSTNAME=0.0.0.0 \
  NODE_ENV=development \
  BROWSER=none \
  BROWSER_OPEN=localhost:5006 \
  yarn start:server-dev > artifacts/chameleon/actual-dev-server.log 2>&1 &

  echo $! > artifacts/chameleon/actual-dev-server.pid
  wait_for_url http://127.0.0.1:3001 "Actual web" 120 || true
  wait_for_url http://127.0.0.1:5006/health "Actual server-dev" 120 || true
}

detect_public_host() {
  if [[ -n "$PUBLIC_HOST" ]]; then
    echo "$PUBLIC_HOST"
    return
  fi

  local metadata_ip=""
  metadata_ip="$(curl -fsS --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
  if [[ -n "$metadata_ip" ]]; then
    echo "$metadata_ip"
    return
  fi

  hostname -I 2>/dev/null | awk '{print $1}'
}

print_summary() {
  local host
  host="$(detect_public_host)"
  [[ -n "$host" ]] || host="<chameleon-floating-ip>"

  cat <<SUMMARY

Bootstrap complete.

Local VM checks:
  Actual frontend:       http://127.0.0.1:3001
  Actual server-dev:     http://127.0.0.1:5006
  Serving API docs:      http://127.0.0.1:8000/docs
  Serving summary:       http://127.0.0.1:8000/monitor/summary
  MLflow:                http://127.0.0.1:5000
  MinIO console:         http://127.0.0.1:9001
  Prometheus:            http://127.0.0.1:9090
  Grafana overview:      http://127.0.0.1:3000/d/actual-ml-system-overview/actual-ml-system-overview

External browser URLs, if the Chameleon security group allows these ports:
  Actual frontend:       http://${host}:3001
  Actual server-dev:     http://${host}:5006
  Serving API docs:      http://${host}:8000/docs
  MLflow:                http://${host}:5000
  MinIO console:         http://${host}:9001
  Prometheus:            http://${host}:9090
  Grafana:               http://${host}:3000

Generated bank import file:
  artifacts/test-data/actual_bank_transactions.qif

Actual UI test flow:
  1. Open Actual frontend.
  2. Choose Create test file.
  3. Open a checking account.
  4. Click Import inside the account page.
  5. Import artifacts/test-data/actual_bank_transactions.qif.
  6. Confirm Category, AI Suggestion, and All Top-3 are populated.

Grafana login:
  admin / admin

If external URLs do not open, update the Chameleon security group to allow:
  TCP 3001, 5006, 8000, 5000, 9001, 9090, 3000
SUMMARY
}

main() {
  ensure_base_packages
  ensure_docker
  ensure_node_and_yarn
  ensure_python_env
  install_yarn_dependencies
  configure_serving_env
  start_serving_stack
  run_demo_traffic
  start_actual_dev_server
  print_summary
}

main
