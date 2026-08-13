# DFlash speculative decoding

DFlash is a **block-diffusion** drafter: instead of chaining single tokens like
`kernel/mtp.py`'s `Drafter`, it drafts a whole block (`block_size`, default 16;
draft depth `block_size-1 = 15`) in ONE parallel forward using mask tokens. The
draft is a small 5-layer qwen3 (4 sliding + 1 full attention) that reuses the
target's `embed_tokens` and `lm_head`, and conditions on the target's hidden
states at `target_layer_ids = [1, 16, 31, 46, 61]` (fused by `fc`, dim `5*H -> H`).

Upstream: <https://github.com/z-lab/dflash> (paper arXiv:2602.06036, MIT).
Drafter model: `z-lab/Qwen3.6-27B-DFlash` (pairs with `Qwen/Qwen3.6-27B`).

## Layers touched

- **L1 `engine.py`** — `enable_hidden_taps(layer_ids)` installs `_TapLayer`
  proxies on the tapped decoder layers; `forward`/`prefill`/`prefill_embeds` set
  `self.last_hidden_taps` (`[B,L,ΣH]`) when enabled, else `None`. Default off →
  MTP/AR paths byte-for-byte unchanged.
- **L2 `mtp.py`** — `Drafter` gained the generic drafter contract methods
  (`update_prefill_context` / `update_context_after_commit` / `commit` /
  `merge_context` / `filter_context`). MTP behaviour is unchanged; its context
  IS the final hidden.
- **L2 `dflash.py`** — `DFlashDrafter` (+ `build_dflash_drafter`) wraps the
  vendored `DFlashDraftModel` behind the generic contract: reads the trunk's
  tapped hiddens from `engine.last_hidden_taps`, drafts a block in one forward,
  and records the committed positions' taps as its cross-attention context
  inside `draft()` (so no separate append/trim on the commit path).
- **L3 `scheduler.py`** — threads an opaque per-row drafter context (`self.ctx`;
  the drafter owns its shape) and delegates its combine/split on join/leave to
  `dr.merge_context` / `dr.filter_context`. The existing verify/accept/take-min
  and **SSM-hybrid rollback** are reused as-is — the SSM-hybrid branch already
  rolls back the qwen3_5 GatedDelta target, replacing DFlash's `_GDNStateCapture`.
- **L4 `hub.py` / L5 `server.py`** — `--dflash <draft-dir>` builds the drafter,
  enables taps, and disables prefix reuse. Concurrent requests batch normally
  (see Batching below).

## Batching (B>1)

DFlash's draft attention is non-standard (a ctx/prop split with a sliding window
and per-row offsets), so batching it the usual way would mean rewriting it over
left-padded `BatchRotatingKVCache` with per-row masks — delicate and easy to get
subtly wrong. Instead we batch only where it pays and keep the draft path on its
validated single-sequence code:

- **Trunk verify stays batched** (the expensive 64-layer forward the scheduler
  already batches). **The 5-layer draft runs per row in a loop** and its outputs
  are stacked — cheap next to the verify, and it reuses `DFlashAttention` as-is.
- **Draft cache is a per-row list**, one entry per batch row (each entry is that
  row's 5-layer cache). `make_cache`/`merge_caches`/`extract_cache_row`/
  `filter_cache` are list ops over rows — no `BatchKVCache`, so the batched
  sliding-window cache is sidestepped entirely.
- **Drafter context is a per-row list too.** On join, an old row's context is
  short (`m+1` committed taps) while a fresh row's is the whole prompt — ragged.
  A per-row list + the per-row draft loop consume each row's context
  independently, so no left-padding/alignment is needed. L3 stops concatenating/
  slicing the context array and calls `dr.merge_context` / `dr.filter_context`
  instead (MTP: array concat/slice; DFlash: list splice).
- take-min commit and the per-row draft-cache invariant (only committed context
  is ever written, so no trim/rollback) both hold per row.

**Tradeoff / TODO — parallelize the draft forward.** The draft model currently
runs B sequential forwards per step (fine for small local concurrency, B≈2–4).
For high concurrency this leaves GPU parallelism on the table; batching it means
porting `DFlashAttention` to left-padded `BatchKVCache`/`BatchRotatingKVCache`
with per-row offsets + masks (and ragged-join alignment). Deferred.

## Run it

```bash
python -m multiplex.server \
  --model Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed \
  --dflash ~/models/Qwen3.6-27B-DFlash
```

Or the standalone reference loop (no scheduler):

```bash
python try_dflash.py \
  --target ~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed \
  --draft  ~/models/Qwen3.6-27B-DFlash \
  --prompt "Write a quicksort in Python." --max-tokens 96
```

## Validated (this machine, Qwen3.6-27B MTPLX target)

- End-to-end through **Hub → Scheduler → Engine**: coherent output, fixed
  depth `D15`, mean accepted ≈ 4 drafts/block. First block accepts ~14.
- Reference `try_dflash.py` on the same prompt: mean_accept ≈ 5.65 (that field is
  `accepted+1`), i.e. ≈ 4.65 accepted drafts. The small gap vs the scheduler path
  is expected batched-forward float drift (see `engine.py` header), not a bug —
  acceptance recovers to 7–11 in predictable regions, confirming ctx threading
  across blocks is correct.
- **MTP regression**: unchanged — dynamic depth active (`D3`, `full=1.000`),
  accept 3/3, ~140 tok/s.

## Limitations (documented, not bugs)

- **Draft forward is not parallelized across rows** — B sequential 5-layer
  forwards per step (see the Batching TODO above). Correct, just not GPU-optimal
  at high concurrency.
- **No prefix-cache reuse of the draft ctx** (`supports_prefix_reuse=False`);
  prefix cache is disabled when `--dflash` is set (trunk-side too, for now).
- **Fixed draft depth, adaptive verify width** (`supports_dynamic_depth=False`,
  `supports_adaptive_verify=True`). The block-diffusion draft is always the
  full trained block (`block_size-1` masks in one parallel forward), but the
  scheduler's depth controller now picks how many of those drafts the trunk
  actually *verifies* (`self.k <= block_size-1`), stepping the verify width down
  in low-acceptance regions and back up when acceptance recovers. It **warms up
  mid-block** (`adaptive_verify_start`, default 7) rather than at the full block:
  starting at 15 would spend a whole short generation stepping the width down
  one level per window before reaching steady state, so short requests would
  never benefit. This is sound
  with no draft-cache rollback because the draft cache only ever stores
  committed context — the drafted block is never written to it (see
  `DFlashAttention`: `update_and_fetch` takes only the ctx K/V). It trims the
  trunk's `1+15`-wide verify forward, not the (unchanged) draft cost, so it
  targets the wasted-verify regression rather than the draft-forward one.
- **Sliding-window drift** past 2048 tokens uses the reference's window-skip; not
  stress-tested for very long generations.

## Registration

The DFlash drafter is **not** a standalone servable model (it needs the target +
`--dflash`), so it is intentionally **not** linked under `~/.mtplx/models`.
Register a DFlash *pairing* only if/when a bundle format is defined.
