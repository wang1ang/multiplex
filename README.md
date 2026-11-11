# multiplex

Local OpenAI-compatible LLM serving for Apple Silicon, built on `mlx-lm`.

Run a local model as a drop-in backend for OpenAI-compatible coding agents.
Point them at `multiplex` instead of a paid API and keep working the same
way, fully offline.

## Features

- OpenAI-compatible `GET /v1/models`, `POST /v1/chat/completions`, and `POST /v1/responses`, with streaming.
- Dynamic batching: new requests can prefill and join a live decode batch.
- MTP speculative decoding when a sidecar is present; falls back to plain autoregressive decoding otherwise.
- Prefix cache with a byte-budgeted in-memory LRU and optional SSD persistence.
- Local model discovery, plus download-by-name from Hugging Face.
- Chat CLI (`try_engine.py`) for local testing and scheduler log inspection.

See `docs/ARCHITECTURE.md` for how the code is layered.

For maintaining the project-local MLX fork, baseline/custom builds, and
incremental Metal-kernel porting from the Qwen challenge, see
[`docs/MLX_KERNEL_MAINTENANCE.md`](docs/MLX_KERNEL_MAINTENANCE.md).

## Install

One command, no clone (needs [uv](https://docs.astral.sh/uv/)):

```bash
uv tool install --python 3.12 git+https://github.com/wang1ang/multiplex
multiplex --model mlx-community/Qwen3-0.6B-4bit   # downloads on first run
```

Or try it without installing anything:

```bash
uvx --python 3.12 --from git+https://github.com/wang1ang/multiplex multiplex
```

This uses the stock `mlx` wheel from PyPI. The repository's own launchers
(`./serve.sh` and friends) build a local MLX fork with one extra kernel fast
path; output is the same either way, the fork is only faster on some shapes.

From a clone:

```bash
pip install -e .            # server/runtime only
pip install -e ".[cli]"     # + the Chat CLI
```

### Linux / CUDA

The same install works on Linux with an NVIDIA GPU and needs no `CUDA_HOME`.
It pulls the CUDA 13 MLX backend and the CUDA headers it compiles kernels
against. The first requests after a fresh install are slow while MLX
compiles its kernels. It caches them for later runs (`MLX_PTX_CACHE_DIR`).

For a CUDA 12 driver, install the `cuda12` extra and swap out the CUDA 13
backend. Both backends write the same `libmlx.so`, so reinstall `mlx-cuda-12`
after the uninstall:

```bash
uv pip install -e ".[cuda12]"
uv pip uninstall mlx-cuda-13
uv pip install --reinstall mlx-cuda-12
```

On NVIDIA GPUs a CUDA-native engine is currently faster. See
[docs/CUDA.md](docs/CUDA.md) for measured numbers and a comparison.

## Quick start

```bash
multiplex --model MODEL_NAME     # installed CLI
./serve.sh --model MODEL_NAME    # from a clone; also wires up pi / Codex
```

Pass a local path, a discovered model's name, a Hugging Face repo id, or a
Hugging Face URL — anything not found locally gets downloaded automatically.
Omit `--model` in an interactive terminal to pick from a numbered list.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a tiny haiku about MLX."}]
  }'
```

`/v1/responses` is also supported, with `previous_response_id` to continue a
conversation.

To use it from aider, Open WebUI, Continue, Cline, the OpenAI SDK and other
clients, see [`docs/CLIENTS.md`](docs/CLIENTS.md).

## Useful server flags

- `-d, --depth N` — initial MTP speculation depth (default `3`); `0` disables it.
- Dynamic MTP has no artificial ceiling; `--no-dynamic-depth` uses the initial depth as fixed.
- `--prefix-cache 8GiB` — resident prefix-cache budget; `0` disables reuse.
- `--wired-limit auto` — pin MLX buffers in RAM up to the GPU's recommended working set so macOS cannot compress or swap the model out; `0` disables.
- `--cache-limit 4GiB` — cap on MLX's cache of freed buffers (it counts toward the wired limit); `auto` keeps MLX's default.
- `--prefix-cache-dir none` — disable SSD-backed prefix cache persistence.
- `--mtp /path/to/sidecar.safetensors` — override automatic MTP discovery.
- `--no-debug` — silence scheduler/request logs.

`serve.sh` also points local coding agents (pi, Codex) at the running server
for the duration, restoring their config on exit. It uses the project-local
`.venv` and custom-built MLX by default; override with `MULTIPLEX_PYTHON` only
for an explicit alternative. Run the server directly with
`.venv/bin/python -m multiplex.server` (same flags) to skip agent wiring.

## Chat CLI

`try_engine.sh`/`try_engine.py` is a local terminal chat client, useful for quick generation
tests and for watching scheduler logs (prefill, JOIN, ADVANCE, MTP acceptance,
prefix-cache hits):

```bash
./try_engine.sh --model MODEL_NAME
```

## Requirements

- macOS on Apple Silicon with an available Metal device (or Linux/CUDA, see above).
- Python 3.10+.

## License

MIT — see [LICENSE](LICENSE).
