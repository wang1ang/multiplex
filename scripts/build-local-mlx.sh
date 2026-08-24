#!/usr/bin/env bash
# Build/install the project-local MLX fork in editable mode.
# No wheel is kept in the repository; the compiled extension is installed into
# the active virtualenv and can be rebuilt after changing third_party/mlx.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$ROOT/third_party/mlx/pyproject.toml" ]]; then
  echo "MLX source missing: $ROOT/third_party/mlx" >&2
  exit 1
fi

export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
cd "$ROOT"
printf 'Building local MLX fork at %s\n' "$(git -C third_party/mlx rev-parse --short HEAD)"
if "$PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON" -m pip install --no-build-isolation --no-deps --editable third_party/mlx
elif command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON" --no-build-isolation --no-deps --editable third_party/mlx
else
  echo "Neither pip nor uv is available for $PYTHON" >&2
  exit 1
fi
"$PYTHON" - <<'PY'
import mlx
import mlx.core as mx
print(f"local MLX import: {mlx.__file__}")
print(f"MLX version: {mx.__version__}")
PY
