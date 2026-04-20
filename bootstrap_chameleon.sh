#!/usr/bin/env bash
set -euo pipefail

# Root-level wrapper so a fresh Chameleon clone only needs one command.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/scripts/chameleon_bootstrap.sh" "$@"
