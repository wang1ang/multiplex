# Using multiplex from other clients

multiplex speaks the OpenAI API, so anything with a "custom OpenAI base URL"
setting can use it. Start the server first:

```bash
multiplex --model MODEL_NAME          # or ./serve.sh from a clone
```

Every client below needs the same three values:

| Setting  | Value                                                        |
|----------|--------------------------------------------------------------|
| Base URL | `http://127.0.0.1:8000/v1`                                   |
| API key  | anything non-empty — the server does not check it            |
| Model    | the id from `curl -s http://127.0.0.1:8000/v1/models`        |

The server runs the one model it was started with and ignores the request's
`model` field, so a wrong model name still works. Clients that validate the
name against `/v1/models` want the real id.

Supported: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/responses`,
streaming (SSE), and tool calls.

Because there is no authentication, keep the default `--host 127.0.0.1`
unless you trust everything that can reach the port.

## pi, Codex

`./serve.sh` from a clone configures these automatically for as long as it
runs: pi gets an extension that reads `/v1/models` on startup, and Codex's
`config.toml` is switched to the server, then restored on exit.

## OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
model = client.models.list().data[0].id
reply = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)
```

`client.responses.create(...)` works too.

## Aider

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=none
aider --model openai/MODEL_ID --no-show-model-warnings
```

The `openai/` prefix tells aider to use the generic OpenAI-compatible route.
Aider has no metadata for a local model and warns about it unless you pass
`--no-show-model-warnings`.

## Open WebUI

Set the connection through environment variables when starting Open WebUI:

```bash
OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=none \
  open-webui serve
```

Or add it in the UI: **Admin Panel → Settings → Connections → OpenAI API**.

If Open WebUI runs in Docker, `127.0.0.1` is the container itself. Use
`http://host.docker.internal:8000/v1` instead. If that connection is refused,
start multiplex with `--host 0.0.0.0`, which also exposes it on your network.

Open WebUI sends extra background requests after each turn (titles, tags,
follow-ups, autocomplete). On a single local model they queue behind your
chat, so disable them with `ENABLE_TITLE_GENERATION=False`,
`ENABLE_TAGS_GENERATION=False`, `ENABLE_FOLLOW_UP_GENERATION=False` and
`ENABLE_AUTOCOMPLETE_GENERATION=False`. `serve-webui.sh` in the repository
does this.

## Continue (VS Code / JetBrains)

In `~/.continue/config.yaml`:

```yaml
name: local
version: 0.0.1
schema: v1
models:
  - name: multiplex
    provider: openai
    model: MODEL_ID
    apiBase: http://127.0.0.1:8000/v1
    apiKey: none
    roles: [chat, edit, apply]
```

## Cline, Roo Code and similar VS Code agents

Choose the **OpenAI Compatible** provider, then enter the base URL, any API
key, and the model id.

## Cursor

Cursor's custom OpenAI base URL is called from Cursor's servers, not from your
machine, so `127.0.0.1` is unreachable from there. Pointing Cursor at multiplex
means exposing the server through a public tunnel. The server has no
authentication, so anyone who finds that URL can use your machine. We don't
recommend it.
