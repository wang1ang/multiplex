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

multiplex_ensure_cli_deps() {
  local python="$1"
  if "$python" -c 'import prompt_toolkit' >/dev/null 2>&1; then
    return 0
  fi
  echo "[multiplex] prompt_toolkit missing; installing CLI dependency..." >&2
  if "$python" -m pip --version >/dev/null 2>&1; then
    "$python" -m pip install 'prompt_toolkit>=3.0' >&2
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python" 'prompt_toolkit>=3.0' >&2
  else
    echo "[multiplex] neither pip nor uv is available" >&2
    return 1
  fi
}
