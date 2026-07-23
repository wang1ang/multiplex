# multiplex

Local OpenAI-compatible LLM serving for Apple Silicon, built on `mlx-lm`.

`multiplex` is aimed at personal agent workloads where requests overlap, prompts
repeat, and long conversations are resent often. It combines a small HTTP server,
a dynamic-batch scheduler, MTP speculative decoding, and prefix-cache reuse.

## Architecture

The stable L1-L4 inference kernel lives in `multiplex/kernel/`. L5 stays in
`multiplex/server.py` because it owns HTTP/OpenAI-compatible protocol concerns.
See `docs/ARCHITECTURE.md` for the layer and dependency boundaries.

## Features

- OpenAI-compatible `GET /v1/models`, `POST /v1/chat/completions`, and `POST /v1/responses`.
- Streaming and non-streaming responses.
- Tool-call parsing for OpenAI-style clients.
- Dynamic batching: new requests can prefill and join a live decode batch.
- MTP speculative decoding when a sidecar is present; pure AR fallback otherwise.
- Prefix cache with in-memory LRU payloads and optional SSD persistence.
- Model discovery under `~/.mtplx/models`.
- Chat CLI (`try_engine.py`) for local testing and scheduler log inspection.

## Install

Core server/runtime:

```bash
pip install -e .
```

Runtime plus the Chat CLI:

```bash
pip install -e ".[cli]"
```

Dependencies are declared in `pyproject.toml`; there is no separate
`requirements.txt`.

## Models

Pass a local model path, a model name discovered under `~/.mtplx/models`, a
Hugging Face repo id, or a Hugging Face model URL. If the command-line model
argument is not found locally, `multiplex` downloads it under `~/.mtplx/models`,
using `--` in place of `/`:

```bash
python -m multiplex.server --model /path/to/model
python -m multiplex.server --model MODEL_NAME
python -m multiplex.server --model org/repo
python -m multiplex.server --model https://huggingface.co/org/repo
```

If `--model` is omitted in an interactive terminal, `multiplex` shows a numbered
model list and Enter selects the first entry. The list includes a few default
downloadable models; entries that are not already local are marked `(需下载)`.
Non-interactive server runs must provide a model when multiple choices are
available.

MTP sidecars are discovered automatically from `mtplx_runtime.json`,
`mtp.safetensors`, or `mtp/weights.safetensors`. Models without a sidecar run
headless in pure autoregressive mode.

## Server

```bash
python -m multiplex.server \
  --model MODEL_NAME \
  --host 127.0.0.1 \
  --port 8000
```

Useful flags:

- `--no-debug`: silence scheduler/request logs.
- `-d, --depth N`: maximum dynamic MTP depth; defaults to `3`. With
  `--no-dynamic-depth`, this becomes the fixed depth. `0` disables speculation.
- `--dynamic-depth`: enabled by default; adapt between D1 and `-d N` from live
  full-depth acceptance. Use `--no-dynamic-depth` for a fixed depth. The
  controller starts at the configured maximum, steps down when recent
  full-depth acceptance is poor, and only retries a higher depth after a
  cooldown. Defaults use a 16-round window with at least 8 observations:
  step down below 50% full-depth acceptance, step up at 80%, and wait 24 rounds
  after a downshift before retrying a higher depth.
- `--prefix-cache-dir auto`: default; persists prefix-cache blocks under
  `~/.cache/multiplex/prefixcache/<model>-<hash>`.
- `--prefix-cache-dir none`: disable SSD-backed prefix cache.
- `--mtp /path/to/sidecar.safetensors`: override automatic MTP discovery.

## API

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a tiny haiku about MLX."}]
  }'
```

Responses API requests are also supported at `/v1/responses`; previous response
items are retained in memory so clients can continue a conversation with
`previous_response_id`.

## Chat CLI

`try_engine.py` is the local Chat CLI. It is useful for quick generation tests
and for watching scheduler logs such as prefill, JOIN, ADVANCE, MTP acceptance,
and prefix-cache behavior.

```bash
python try_engine.py --model MODEL_NAME  # dynamic D1..D3 by default
python try_engine.py --model MODEL_NAME -d 3 --prompt-file prompts.jsonl
python try_engine.py --model MODEL_NAME -d 2  # dynamic D1..D2
python try_engine.py --model MODEL_NAME --no-dynamic-depth  # fixed D3
python try_engine.py --model MODEL_NAME --no-debug
```

The UI has a generated-output pane, a scheduler-log pane, and a fixed input box.
The Chat CLI defaults to dynamic depth with a maximum of D3. `-d N` changes the
maximum, while `--no-dynamic-depth` selects a fixed depth.
`--prompt TEXT` submits an initial prompt directly. `--prompt-file PATH` accepts
plain text or the first `prompt` field in a JSON/JSONL file.

These defaults are shared by the HTTP server, `Hub`, `Scheduler`, and the
vision CLI. If no MTP sidecar is available, all entry points still fall back to
pure autoregressive decoding.

## Requirements

- macOS on Apple Silicon with an available Metal device.
- Python 3.10+.
- A local MLX / `mlx-lm` compatible model directory.

## Status

This is an active local-serving project, not a general-purpose hosted inference
stack. The lower layers are intentionally small and geared toward debugging
scheduler behavior on one Apple Silicon machine.
