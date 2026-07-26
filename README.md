# multiplex

Local OpenAI-compatible LLM serving for Apple Silicon, built on `mlx-lm`.

Run a local model as a drop-in backend for OpenAI-compatible coding agents
(Codex, pi, ...) — point them at `multiplex` instead of a paid API and keep
working the same way, fully offline.

## Features

- OpenAI-compatible `GET /v1/models`, `POST /v1/chat/completions`, and `POST /v1/responses`, with streaming.
- Dynamic batching: new requests can prefill and join a live decode batch.
- MTP speculative decoding when a sidecar is present; falls back to plain autoregressive decoding otherwise.
- Prefix cache with a byte-budgeted in-memory LRU and optional SSD persistence.
- Model discovery under `~/.mtplx/models`, plus download-by-name from Hugging Face.
- Chat CLI (`try_engine.py`) for local testing and scheduler log inspection.

See `docs/ARCHITECTURE.md` for how the code is layered.

## Install

```bash
pip install -e .            # server/runtime only
pip install -e ".[cli]"     # + the Chat CLI
```

### Linux / CUDA

Set `CUDA_HOME` before the first run — MLX's CUDA backend JIT-compiles
kernels and fails without it (use `-e ".[cuda12]"` for a CUDA 12 driver):

```bash
export CUDA_HOME="$(python -c 'import nvidia, pathlib
print(pathlib.Path(list(nvidia.__path__)[0]) / "cu13")')"
```

## Quick start

```bash
./serve.sh --model MODEL_NAME
```

Pass a local path, a name under `~/.mtplx/models`, a Hugging Face repo id, or
a Hugging Face URL — anything not found locally gets downloaded automatically.
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

## Useful server flags

- `-d, --depth N` — max MTP speculation depth (default `3`); `0` disables it.
- `--no-dynamic-depth` — use a fixed depth instead of adapting to live acceptance.
- `--prefix-cache 4GiB` — resident prefix-cache budget; `0` disables reuse.
- `--prefix-cache-dir none` — disable SSD-backed prefix cache persistence.
- `--mtp /path/to/sidecar.safetensors` — override automatic MTP discovery.
- `--no-debug` — silence scheduler/request logs.

`serve.sh` also points local coding agents (pi, Codex) at the running server
for the duration, restoring their config on exit. Run the server directly with
`python -m multiplex.server` (same flags) to skip that.

## Chat CLI

`try_engine.py` is a local terminal chat client, useful for quick generation
tests and for watching scheduler logs (prefill, JOIN, ADVANCE, MTP acceptance,
prefix-cache hits):

```bash
python try_engine.py --model MODEL_NAME
```

## Requirements

- macOS on Apple Silicon with an available Metal device (or Linux/CUDA, see above).
- Python 3.10+.

## License

MIT — see [LICENSE](LICENSE).
