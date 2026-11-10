# Changelog

## 0.1.0 — 2026-11-10

First tagged release.

### Serving
- OpenAI-compatible `GET /v1/models`, `POST /v1/chat/completions` and
  `POST /v1/responses` (with `previous_response_id`), streaming over SSE.
- Structured tool calls, streaming and non-streaming.
- Reasoning returned as `reasoning_content`.
- Dynamic batching: new requests prefill and join a live decode batch.
- Models given by local path, discovered name, Hugging Face repo id or URL;
  missing ones are downloaded.
- `uv tool install git+https://github.com/wang1ang/multiplex` gives a
  `multiplex` command.
- `./serve.sh` points pi and Codex at the server while it runs.

### Speculative decoding
- Batched MTP speculative decoding from a sidecar, with dynamic depth.
- DFlash block-diffusion drafter for Qwen3.6-27B targets.
- Context-copy (prompt lookup) blocks for grounded re-emission, such as code
  edits.

### Memory and caching
- Prefix cache with a byte-budgeted in-memory LRU (default 8 GiB) and SSD
  persistence; sparse SSM snapshots on hybrid models.
- `--wired-limit` / `--cache-limit` keep the model resident under memory
  pressure.

### Kernels
- Fused GDN conv-norm and decode-step kernels.
- GDN-only replay on rollback.
