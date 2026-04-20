#!/usr/bin/env bash
set -euo pipefail

# Serving-folder compatibility wrapper. The canonical bootstrap lives at repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
exec bash "$ROOT_DIR/scripts/chameleon_bootstrap.sh" --serving-only "$@"
