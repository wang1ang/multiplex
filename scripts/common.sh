#!/usr/bin/env bash
# Shared launcher helpers. Source this file from project entrypoints.

multiplex_ensure_mlx() {
  local root="$1"
  local python="${MULTIPLEX_PYTHON:-$root/.venv/bin/python}"

  if [ -x "$python" ] && "$python" -c 'import mlx' >/dev/null 2>&1; then
    printf '%s\n' "$python"
    return 0
  fi

  if [ -n "${MULTIPLEX_PYTHON:-}" ]; then
    echo "MULTIPLEX_PYTHON does not provide mlx: $python" >&2
    return 1
  fi

  echo "[multiplex] local MLX build missing; building..." >&2
  "$root/scripts/build-local-mlx.sh" >&2
  [ -x "$root/.venv/bin/python" ] || {
    echo "[multiplex] local Python/MLX build was not created" >&2
    return 1
  }
  printf '%s\n' "$root/.venv/bin/python"
}
