# multiplex

Local LLM serving for Apple Silicon.

`multiplex` is built for personal agent workloads: fast local generation,
OpenAI-compatible endpoints, streaming responses, tool calling, dynamic batching,
and long-conversation reuse.

The goal is simple: make a local model feel responsive even when an agent sends
several overlapping requests or repeatedly resends a long conversation.

## Highlights

- **OpenAI-compatible API** for `/v1/responses`, `/v1/chat/completions`, and `/v1/models`.
- **Streaming output** for chat and agent clients.
- **Tool-call support** for clients that expect structured OpenAI-style tool calls.
- **Dynamic batching** so overlapping requests can run together.
- **Speculative decoding** when the model supports it, with pure AR fallback when it does not.
- **Prefix reuse** for long-running conversations and retries.
- **Chat CLI** for local testing.

## Quick Start

Install:

```bash
pip install -r requirements.txt
pip install -e ".[cli]"
```

Start the server:

```bash
python -m multiplex.server --model /path/to/model --host 127.0.0.1 --port 8000
```

Or use a model name discovered under `~/.mtplx/models`:

```bash
python -m multiplex.server --model MODEL_NAME
```

Try the chat CLI:

```bash
python try_engine.py --model /path/to/model
```

## API

Available endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

Streaming and non-streaming responses are supported.

Example:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a tiny haiku about MLX."}]
  }'
```

## Requirements

- macOS with Apple Silicon and an available Metal device.
- Python 3.10+.
- `mlx` and `mlx-lm`.
- A local MLX model directory.

## Status

`multiplex` is an active local-serving project focused on Apple Silicon and
agent-style workflows.
