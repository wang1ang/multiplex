#!/usr/bin/env bash
# Keep the stock wheel and the project-local MLX build side by side.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
UV="${UV:-$(command -v uv || true)}"
BASELINE_DIR="$ROOT/.build/mlx-baseline"
CUSTOM_DIR="$ROOT/.build/mlx-custom"

[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON" >&2; exit 1; }
[[ -n "$UV" ]] || { echo "uv not found" >&2; exit 1; }
mkdir -p "$BASELINE_DIR" "$CUSTOM_DIR"

SITE="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
KERNEL="$ROOT/third_party/mlx/mlx/backend/metal/kernels/quantized.h"
BACKUP="$(mktemp)"
trap 'cp "$BACKUP" "$KERNEL"; rm -f "$BACKUP"' EXIT
cp "$KERNEL" "$BACKUP"

# Build the exact pre-edit source as the baseline.
git -C "$ROOT/third_party/mlx" show HEAD:mlx/backend/metal/kernels/quantized.h > "$KERNEL"
"$ROOT/scripts/build-local-mlx.sh"
rm -rf "$BASELINE_DIR/mlx" "$BASELINE_DIR"/mlx-*.dist-info
cp -R "$ROOT/third_party/mlx/python/mlx" "$BASELINE_DIR/mlx"
for d in "$SITE"/mlx-*.dist-info; do
  [[ -d "$d" ]] && cp -R "$d" "$BASELINE_DIR/"
done

# Restore the edited source, build it, then snapshot the custom package.
cp "$BACKUP" "$KERNEL"
"$ROOT/scripts/build-local-mlx.sh"
rm -rf "$CUSTOM_DIR/mlx" "$CUSTOM_DIR"/mlx-*.dist-info
cp -R "$ROOT/third_party/mlx/python/mlx" "$CUSTOM_DIR/mlx"
for d in "$SITE"/mlx-*.dist-info; do
  [[ -d "$d" ]] && cp -R "$d" "$CUSTOM_DIR/"
done
printf 'baseline: %s\ncustom: %s\n' "$BASELINE_DIR" "$CUSTOM_DIR"
