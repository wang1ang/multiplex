#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/scripts/common.sh"
PYTHON="$(multiplex_ensure_mlx "$HERE")"
multiplex_ensure_cli_deps "$PYTHON"
exec "$PYTHON" "$HERE/try_engine.py" "$@"
