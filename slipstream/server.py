"""L5 — HTTP / OpenAI-compatible layer. PROTOCOL TRANSLATION ONLY.

Translates HTTP requests <-> an internal (prompt_ids, params) request and back
to JSON / SSE. It does NOT schedule, batch, or manage cache — those belong to
L3/L4 (not written yet). Until then a single lock serializes requests (TEMP,
replace with L3/L4).

Endpoints (each a different wire format, all translating to the same internal
generate call):
  * POST /v1/chat/completions   — classic Chat Completions (+ SSE)
  * POST /v1/responses          — OpenAI Responses API (what Codex uses; +SSE)
  * GET  /v1/models

Sync stdlib http.server: the engine is a sync generator, so a sync server pairs
with it directly (no async bridge).
"""

from __future__ import annotations

import os
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx

from .engine import Engine
from .mtp import Drafter, speculative_generate_batch


# --- core: one internal generate bridge every endpoint shares -----------------
class _Backend:
    """Owns the engine + drafter. Turns (prompt_ids, params) into a token stream.

    TEMP serialization: the server is single-threaded (HTTPServer), so requests
    are processed one at a time — that IS the temporary queue. MLX's GPU stream
    is thread-bound, which also forces single-threaded here. L3/L4 replaces this
    with real batched scheduling.
    """

    def __init__(self, model_path: str, mtp_path: str, bits: int | None = 4):
        self.eng = Engine(model_path)
        self.drafter = Drafter(self.eng, mtp_path, bits=bits)
        self.model_id = model_path.rstrip("/").split("/")[-1]

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        return self.eng.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    def stream_text(self, prompt_ids, max_tokens, k=1):
        """Yield decoded text deltas for one request."""
        produced, shown = [], ""
        for _row_ids, step in speculative_generate_batch(
            self.eng, self.drafter, [prompt_ids], max_tokens=max_tokens, k=k
        ):
            produced.extend(step[0])   # single request here -> always row 0
            full = self.eng.decode(produced)
            if full != shown:
                yield full[len(shown):]
                shown = full


def _hex(prefix):
    return prefix + uuid.uuid4().hex


def _messages_from_chat(body):
    return list(body.get("messages", []))


def _messages_from_responses(body):
    """Responses: optional `instructions` (system) + `input` (str or items)."""
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    inp = body.get("input")
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and item.get("type") in ("message", None):
                c = item.get("content")
                if isinstance(c, list):  # content parts -> join text
                    c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                if isinstance(c, str):
                    msgs.append({"role": item.get("role", "user"), "content": c})
    return msgs


# --- SSE encoders -------------------------------------------------------------
def _sse(data, event=None):
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data)}\n\n"


def _chat_stream(backend, prompt_ids, max_tokens):
    rid, created = _hex("chatcmpl-"), int(time.time())

    def chunk(delta, finish=None):
        return _sse({
            "id": rid, "object": "chat.completion.chunk", "created": created,
            "model": backend.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    yield chunk({"role": "assistant"})
    for text in backend.stream_text(prompt_ids, max_tokens):
        yield chunk({"content": text})
    yield chunk({}, finish="stop")
    yield "data: [DONE]\n\n"


def _responses_stream(backend, prompt_ids, max_tokens):
    rid, mid = _hex("resp_"), _hex("msg_")
    base = {"id": rid, "object": "response", "model": backend.model_id}

    yield _sse({"type": "response.created", "response": {**base, "status": "in_progress"}},
               "response.created")
    yield _sse({"type": "response.output_item.added", "item":
                {"id": mid, "type": "message", "role": "assistant", "content": [],
                 "status": "in_progress"}}, "response.output_item.added")

    full = ""
    for text in backend.stream_text(prompt_ids, max_tokens):
        full += text
        yield _sse({"type": "response.output_text.delta", "item_id": mid, "delta": text},
                   "response.output_text.delta")

    yield _sse({"type": "response.output_text.done", "item_id": mid, "text": full},
               "response.output_text.done")
    yield _sse({"type": "response.completed", "response": {
        **base, "status": "completed", "created_at": int(time.time()),
        "output": [{"id": mid, "type": "message", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": full, "annotations": []}]}],
    }}, "response.completed")


# --- non-stream bodies --------------------------------------------------------
def _chat_body(backend, text):
    return {
        "id": _hex("chatcmpl-"), "object": "chat.completion",
        "created": int(time.time()), "model": backend.model_id,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
    }


def _responses_body(backend, text):
    mid = _hex("msg_")
    return {
        "id": _hex("resp_"), "object": "response", "created_at": int(time.time()),
        "status": "completed", "model": backend.model_id,
        "output": [{"id": mid, "type": "message", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
    }


# --- HTTP handler -------------------------------------------------------------
def make_handler(backend: _Backend):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _sse_stream(self, gen):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ev in gen:
                self.wfile.write(ev.encode())
                self.wfile.flush()

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": backend.model_id, "object": "model", "owned_by": "local"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            path = self.path.rstrip("/")
            stream = bool(body.get("stream", False))
            max_tokens = int(body.get("max_output_tokens")
                             or body.get("max_tokens") or 2048)

            if path == "/v1/chat/completions":
                msgs = _messages_from_chat(body)
                ids = backend.prompt_ids(msgs)
                if stream:
                    self._sse_stream(_chat_stream(backend, ids, max_tokens))
                else:
                    text = "".join(backend.stream_text(ids, max_tokens))
                    self._json(200, _chat_body(backend, text))
            elif path == "/v1/responses":
                msgs = _messages_from_responses(body)
                ids = backend.prompt_ids(msgs)
                if stream:
                    self._sse_stream(_responses_stream(backend, ids, max_tokens))
                else:
                    text = "".join(backend.stream_text(ids, max_tokens))
                    self._json(200, _responses_body(backend, text))
            else:
                self._json(404, {"error": "not found"})

    return Handler


def serve(model_path: str, mtp_path: str, host="127.0.0.1", port=8000, bits=4):
    backend = _Backend(model_path, mtp_path, bits=bits)
    httpd = HTTPServer((host, port), make_handler(backend))
    print(f"[serving {backend.model_id} on http://{host}:{port}  "
          f"(/v1/chat/completions, /v1/responses, /v1/models)]")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse

    MODEL = os.path.expanduser("~/.mtplx/models/Agents-A1-MTPLX")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--mtp", default=MODEL + "/mtp.safetensors")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    serve(args.model, args.mtp, host=args.host, port=args.port)
