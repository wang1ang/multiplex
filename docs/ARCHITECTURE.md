# Architecture

```text
L5  server.py               OpenAI-compatible HTTP / SSE / JSON translation
L4  kernel/hub.py           many caller threads -> one engine thread
L3  kernel/scheduler.py     prefill, merge, decode step, cancel, prefix cache
L2  kernel/mtp.py           MTP sidecar, draft generation, norm/quant loading
L1  kernel/engine.py        batched forward, logits, cache clone/filter/restore
    kernel/prefixcache/     L3 prefix-cache policy, state adapter, disk store
```

`multiplex/kernel/` (L1-L4) is meant to stay boring and stable. Prefer making
protocol, client-compatibility, and parsing changes at the edge instead —
`bridge/`, `registry.py`, tests, docs.

Dependency rule: kernel code depends downward only, never on `bridge/`,
`registry.py`, or HTTP/wire concerns. The one exception is L4's `Hub`, which
calls `bridge.normalize_messages_for_template` and `bridge.ThinkingParser` to
prep messages and parse thinking blocks. Everything OpenAI-specific (JSON
shapes, SSE events, HTTP status) lives in L5 and must not leak below it.
