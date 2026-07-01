"""L3 — dynamic-batch scheduler ("live batch").

A running batch that requests can join and leave mid-flight:
  * 入 (join): a new request is prefilled in chunks (interleaved with decode so
    it never blocks the running rows), then merged into the batch.
  * 推 (advance): each step speculatively decodes every live row one round.
    Speculation IS the generation — there is no separate AR path.
  * 出 (leave): a row that hits EOS is filtered out and its cache released.

One step does BOTH: advance the running batch by one speculative round AND feed
one prefill chunk of a joining request. The two are separate forwards (different
shapes/caches) run back to back — never merged.

Single-threaded: MLX's GPU stream is thread-bound and the model is shared, so
the scheduler drives everything from one loop.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import mlx.core as mx

from .engine import Engine, BatchState
from .mtp import Drafter


@dataclass
class _Req:
    rid: int                      # stable request id
    prompt: list[int]
    max_tokens: int
    out: list[int] = field(default_factory=list)   # generated tokens
    done: bool = False


@dataclass
class _Prefill:
    """A request being prefilled in chunks before it joins the batch."""
    req: _Req
    state: BatchState             # its own single-row cache, grown chunk by chunk
    pos: int = 0                  # how many prompt tokens prefilled so far


class Scheduler:
    def __init__(self, engine: Engine, drafter: Drafter, *, k=1, chunk=512, debug=False):
        self.eng = engine
        self.dr = drafter
        self.k = k
        self.chunk = chunk
        self.eos = engine.eos_token_ids

        # running batch
        self.state: BatchState | None = None
        self.h = None                 # [B,1,H] each live row's next-input hidden
        self.primary = None           # [B] each live row's next token
        self.dcache = drafter.make_cache()
        self.rows: list[_Req] = []    # live rows -> request (row i == rows[i])

        self.waiting: list[_Req] = []     # queued, not yet prefilling
        self.prefilling: _Prefill | None = None   # at most one in flight (single-user)
        self._next_id = 0
        self.debug = debug
        self._t = 0                       # step counter for logs

    def _log(self, msg: str):
        if self.debug:
            print(f"[sched t={self._t}] {msg}", file=sys.stderr, flush=True)

    # --- public: request lifecycle ------------------------------------------
    def add(self, prompt: list[int], max_tokens: int) -> int:
        r = _Req(self._next_id, list(prompt), max_tokens)
        self._next_id += 1
        self.waiting.append(r)
        self._log(f"ADD req{r.rid} (prompt={len(r.prompt)} tok, max={max_tokens}); "
                  f"waiting={len(self.waiting)}")
        return r.rid

    def active(self) -> bool:
        return bool(self.rows or self.waiting or self.prefilling)

    # --- one scheduler step: advance running batch + feed one prefill chunk --
    def step(self) -> list[tuple[int, list[int]]]:
        """Advance one round. Returns [(rid, new_tokens), ...] for rows that
        produced tokens this step (a joining row's first token is included)."""
        self._t += 1
        emitted: list[tuple[int, list[int]]] = []

        # 推 + 出: one speculative round over the running batch
        if self.rows:
            emitted = self._advance()

        # 入: feed one prefill chunk; when a request finishes prefill, merge it.
        # A row that just joined emits its (already-sampled) first token now.
        joined = self._advance_prefill()
        if joined is not None:
            rid, first = joined
            emitted.append((rid, [first]))

        return emitted

    def run(self):
        """Drive to completion, yielding (rid, new_tokens) per step."""
        while self.active():
            for rid, toks in self.step():
                yield rid, toks

    # --- 入: chunked prefill + merge ----------------------------------------
    def _advance_prefill(self):
        """Feed one prefill chunk. Returns (rid, first_token) if a request just
        finished prefill and joined the batch this step, else None."""
        # start a new prefill if none in flight and something is waiting
        if self.prefilling is None and self.waiting:
            req = self.waiting.pop(0)
            self.prefilling = _Prefill(req=req, state=None, pos=0)
            self._log(f"PREFILL start req{req.rid} (prompt={len(req.prompt)} tok)")

        pf = self.prefilling
        if pf is None:
            return None

        # feed one chunk
        end = min(pf.pos + self.chunk, len(pf.req.prompt))
        chunk = pf.req.prompt[pf.pos:end]
        if pf.state is None:
            pf.state, hid = self.eng.prefill([chunk])
        else:
            hid = self.eng.forward(pf.state, mx.array([chunk]))
        self._log(f"PREFILL req{pf.req.rid} chunk [{pf.pos}:{end}]/{len(pf.req.prompt)}")
        pf.pos = end

        if pf.pos < len(pf.req.prompt):
            return None  # more chunks next step

        # prefill done: sample first token, merge into the running batch
        first = int(mx.argmax(self.eng.logits(hid)[0, -1]))
        pf.req.out.append(first)
        last_h = hid[:, -1:, :]                       # [1,1,H]
        self._merge_in(pf.req, pf.state, last_h, first)
        rid = pf.req.rid
        self.prefilling = None
        self._log(f"JOIN req{rid} -> batch (now {len(self.rows)} rows: "
                  f"{[r.rid for r in self.rows]}), first_tok={first}")
        return (rid, first)

    def _merge_in(self, req, rstate, last_h, first):
        if self.state is None:                        # empty batch -> becomes the batch
            self.state = rstate
            self.h = last_h
            self.primary = mx.array([first])
            self.rows = [req]
            self.dcache = self.dr.make_cache()
            return
        # merge caches (attention left-pad, SSM stack) and per-row inputs
        self.state = self.eng.merge_states([self.state, rstate])
        self.h = mx.concatenate([self.h, last_h], axis=0)
        self.primary = mx.concatenate([self.primary, mx.array([first])], axis=0)
        self.rows.append(req)
        # draft cache: the new row has no draft history yet; reset shared dcache
        # (drafts are recomputed each round from trunk hidden, so a fresh cache
        # for the whole batch is correct).
        self.dcache = self.dr.make_cache()

    # --- 推 + 出: one speculative round over the running batch ---------------
    def _advance(self) -> list[tuple[int, list[int]]]:
        eng, dr, k, eos = self.eng, self.dr, self.k, self.eos
        state, h, primary, rows = self.state, self.h, self.primary, self.rows
        B = len(rows)

        drafts = dr.draft(h, primary, k, self.dcache)          # [B,k]
        draft_ids = [[int(x) for x in drafts[i]] for i in range(B)]

        snap = eng.snapshot_ssm(state)
        lengths_before = list(state.lengths)
        verify_in = mx.array([[int(primary[i])] + draft_ids[i] for i in range(B)])
        vhidden = eng.forward(state, verify_in)
        trunk_pred = mx.argmax(eng.logits(vhidden), axis=-1)   # [B,k+1]

        accs = []
        for i in range(B):
            a = 0
            for j in range(k):
                if draft_ids[i][j] == int(trunk_pred[i, j]):
                    a += 1
                else:
                    break
            accs.append(a)
        m = min(accs)
        self._log(f"ADVANCE {B} rows {[r.rid for r in rows]} | accept per-row={accs} "
                  f"min={m} -> commit {m + 1} tok/row")

        emitted, finished = [], []
        for i in range(B):
            toks = draft_ids[i][:m] + [int(trunk_pred[i, m])]
            for j, t in enumerate(toks):
                if t in eos or len(rows[i].out) + j + 1 >= rows[i].max_tokens:
                    toks = toks[: j + 1]
                    finished.append(i)
                    break
            rows[i].out.extend(toks)
            emitted.append((rows[i].rid, toks))

        # next inputs / repair
        primary = trunk_pred[:, m]
        if m == k:
            h = vhidden[:, -1:, :]
        else:
            eng.restore_ssm(state, snap)
            eng.trim_attention(state, k - m)
            state.lengths = list(lengths_before)
            commit_in = mx.array(
                [[int(verify_in[i, 0])] + draft_ids[i][:m] for i in range(B)]
            )
            h = eng.forward(state, commit_in)[:, -1:, :]

        self.state, self.h, self.primary = state, h, primary

        # 出: drop finished rows
        if finished:
            self._log(f"EXIT rows {[rows[i].rid for i in finished]} (EOS/max)")
            keep = [i for i in range(B) if i not in finished]
            if keep:
                eng.filter(self.state, keep)
                dr.filter_cache(self.dcache, keep)
                self.primary = self.primary[mx.array(keep)]
                self.h = self.h[mx.array(keep)]
                self.rows = [rows[i] for i in keep]
            else:
                self.state = self.h = self.primary = None
                self.rows = []
                self.dcache = dr.make_cache()

        return emitted
