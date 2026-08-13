# Architecture

```text
L5  server.py               OpenAI-compatible HTTP / SSE / JSON translation
L4  kernel/hub.py           many caller threads -> one engine thread
L3  kernel/scheduler.py     prefill, merge, decode step, cancel, prefix cache
L2  kernel/mtp.py           drafter contract + MTP impl (sidecar, norm/quant)
    kernel/dflash.py        DFlash drafter impl (block-diffusion, hidden taps)
L1  kernel/engine.py        batched forward, logits, cache clone/filter/restore,
                            optional intermediate-hidden taps
    kernel/prefixcache/     L3 prefix-cache policy, state adapter, disk store
```

L2 is the **speculation layer**: L3 talks to a generic drafter contract
(`make_cache` / `cache_size` / `draft` / `make_context` / … — see
`docs/DRAFTER_INTERFACE.md`), and MTP is one implementation, DFlash another. L3
stays the KV-cache manager for both the trunk and the draft cache — it speaks
shape-agnostic verbs, not MTP-specific ones. A model with no head/drafter runs
pure AR.

`multiplex/kernel/` (L1-L4) is meant to stay boring and stable. Prefer making
protocol, client-compatibility, and parsing changes at the edge instead —
`bridge/`, `registry.py`, tests, docs.

Dependency rule: kernel code depends downward only, never on `bridge/`,
`registry.py`, or HTTP/wire concerns. The one exception is L4's `Hub`, which
calls `bridge.normalize_messages_for_template` and `bridge.ThinkingParser` to
prep messages and parse thinking blocks. Everything OpenAI-specific (JSON
shapes, SSE events, HTTP status) lives in L5 and must not leak below it.

## Layer boundaries

Each layer owns one job; the lines between them are what keep the kernel boring.

- **L5 `server.py`** — the ONLY layer that knows the OpenAI wire format: HTTP
  routing, SSE framing, JSON request/response shapes, status codes. It renders
  what L4 streams and never touches tokens, caches, or scheduling.
- **L4 `hub.py`** — the funnel. Many HTTP threads submit here; one engine thread
  runs the model (MLX's GPU stream is thread-bound). Owns chat-template
  rendering, tokenization, the `<think>` split, request admission (incl. keeping
  a non-batchable drafter single-stream), and picks the drafter implementation.
  It does not decode or manage caches.
- **L3 `scheduler.py`** — the batch + **KV-cache manager**. Owns prefill,
  merge/join, the speculative decode step (verify / accept-longest-prefix /
  take-min), cancellation, and the prefix cache. See the cache rule below.
- **L2 `mtp.py` / `dflash.py`** — the speculation layer: *how* to draft, and the
  *shape* of the draft cache. Implements the drafter contract; owns the draft
  model/head, its cache representation (MTP: 1+ KVCache; DFlash: 5 caches incl.
  sliding windows), and which trunk hidden it reads (final vs taps). It does not
  decide batch lifecycle events.
- **L1 `engine.py`** — correct forward passes and the mechanics of the trunk
  cache (clone / filter / restore / trim, optional hidden taps). It never
  samples or schedules.

### The cache rule (subtle, easy to miss)

**L3 manages every KV cache's lifecycle — the trunk cache AND L2's draft cache.**
MTP (and DFlash) each carry their own draft KV cache, but *when* the lifecycle
actions happen — create, merge on join, split/filter on leave, trim after a
verify, snapshot for the prefix cache — is decided by L3, because only L3 sees
those batch events. L2 owns the cache's *representation and mechanics*; L3 drives
its *lifecycle*.

Which actions actually fire depends on the drafter, because L3 speaks the
lifecycle as intent, not mechanism. MTP writes speculative tokens into its draft
cache, so after a verify L3 trims the rejected tail (`trim_to`). **DFlash writes
only committed context into its cache (the drafted block never enters it), so the
trim is a no-op and nothing is rolled back** — DFlash still HAS a cache (5 layers)
and still needs it; it just never accumulates anything to undo. (DFlash v1 is
also single-stream, so merge/filter aren't exercised yet either.)

Because L3 speaks these as **shape-agnostic verbs**
(`make_cache` / `cache_size` / `trim_to` / `extract_row` / `merge` / `filter` /
`snapshot`), never MTP-specific ones (no `dcache[0]`, no “base+1”), one manager
drives both MTP's single-layer cache and DFlash's five. Only the draft
*algorithm* and the *model-compute* that grows the cache live in L2. See
`docs/DRAFTER_INTERFACE.md` for the full contract and migration map.
