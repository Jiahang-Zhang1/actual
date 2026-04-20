#!/usr/bin/env bash
set -euo pipefail

# Build the modified Actual Budget web/sync image used by Kubernetes.
# The image contains Node 22, production dependencies, compiled web assets,
# and the sync server that serves the app on port 5006.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

MODE="${1:-load}"
IMAGE="${IMAGE:-actual-smartcat/actual-sync}"
TAG="${TAG:-serving-$(git rev-parse --short HEAD)}"
PLATFORM="${PLATFORM:-linux/amd64}"
REVISION="$(git rev-parse HEAD)"
CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

safe_image_name="${IMAGE//\//_}"
safe_platform="${PLATFORM//\//_}"
TAR_PATH="${TAR_PATH:-artifacts/docker/${safe_image_name}-${TAG}-${safe_platform}.tar}"

usage() {
  cat <<EOF
Usage:
  IMAGE=ghcr.io/<org>/actual-sync TAG=serving-<sha> $0 load
  IMAGE=ghcr.io/<org>/actual-sync TAG=serving-<sha> $0 push
  IMAGE=actual-smartcat/actual-sync TAG=serving-<sha> $0 tar

Modes:
  load  Build and load the image into the local Docker daemon.
  push  Build and push the image to IMAGE:TAG.
  tar   Build a Docker image tarball at TAR_PATH for offline node import.

Defaults:
  IMAGE=${IMAGE}
  TAG=${TAG}
  PLATFORM=${PLATFORM}
  TAR_PATH=${TAR_PATH}
EOF
}

if [[ "${MODE}" == "--help" || "${MODE}" == "-h" ]]; then
  usage
  exit 0
fi

build_args=(
  docker buildx build
  --platform "${PLATFORM}"
  --file Dockerfile.sync-k8s
  --tag "${IMAGE}:${TAG}"
  --build-arg "BUILD_REVISION=${REVISION}"
  --build-arg "BUILD_CREATED=${CREATED}"
)

case "${MODE}" in
  load)
    build_args+=(--load .)
    ;;
  push)
    build_args+=(--push .)
    ;;
  tar)
    mkdir -p "$(dirname "${TAR_PATH}")"
    build_args+=(--output "type=docker,dest=${TAR_PATH}" .)
    ;;
  *)
    usage
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

echo "Building ${IMAGE}:${TAG} for ${PLATFORM} using Dockerfile.sync-k8s"
"${build_args[@]}"

if [[ "${MODE}" == "tar" ]]; then
  echo "Wrote image tarball to ${TAR_PATH}"
fi
