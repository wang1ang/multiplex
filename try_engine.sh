#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MULTIPLEX_PYTHON:-$HERE/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${MULTIPLEX_PYTHON:-python3}"
fi
exec "$PYTHON" "$HERE/try_engine.py" "$@"
