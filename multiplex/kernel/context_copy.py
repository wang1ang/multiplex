"""Context-copy (prompt-lookup) block speculation.

Port of MTPLX's context-copy lane (mtplx/context_copy.py, RAMP left out). When
the tail of prompt+output matches an n-gram that occurs in the PROMPT, the
prompt's continuation is proposed verbatim as a block and verified in one trunk
forward through the scheduler's normal accept rule, so the output law is the
same as an MTP round (greedy: bit-exact). Wins on grounded re-emission (code
edits, file rewrites, RAG quotes), where an MTP round commits at most k+1
tokens but a copy block can commit up to 32.

Only the prompt is indexed: matches into the model's own output (self
repetition) predict poorly and cost more to verify than they commit.

Per-request policy (same constants as upstream):
  * block length by backward match extension: a longer suffix match earns a
    longer block (BLOCK_LADDER), so a weak match only risks a short verify;
  * probation: blocks stay <= PROBATION_K until the acceptance EMA shows this
    content pays, then open to BLOCK_K;
  * suspension: when the EMA drops below 0.35 the lane pauses for ``backoff``
    tokens, doubling on each suspension (cap 4096), reset by a paying round.
"""
from __future__ import annotations

NG_MIN = 6
NG_MAX = 10
BLOCK_K = 24
PROBATION_K = 8
MAX_CANDIDATES = 32
BLOCK_LADDER = (8, 12, 16, 24, 32)


class ContextCopy:
    """Per-request proposer + acceptance policy over one prompt."""

    def __init__(self, prompt: list[int]):
        self.prompt = prompt
        # NG_MIN-gram -> end positions in the prompt (= continuation start).
        self.grams: dict[tuple, list[int]] = {}
        for e in range(NG_MIN, len(prompt)):      # e == len(prompt): nothing to copy
            self.grams.setdefault(tuple(prompt[e - NG_MIN:e]), []).append(e)
        self.ema, self.seen = 0.5, 0
        self.suspend_until, self.backoff = 0, 64

    def _match(self, tail: list[int]):
        """Best ``(continuation_pos, extension)`` for the history ``tail``.

        Extension = how many tokens beyond NG_MIN the match also runs
        backwards; longest wins, most recent breaks ties.
        """
        cands = self.grams.get(tuple(tail[-NG_MIN:]))
        if not cands:
            return None, -1
        best_pos, best_ext = None, -1
        for pos in reversed(cands[-MAX_CANDIDATES:]):
            ext = 0
            while (ext < NG_MAX - NG_MIN
                   and ext + NG_MIN < len(tail)
                   and pos - NG_MIN - 1 - ext >= 0
                   and self.prompt[pos - NG_MIN - 1 - ext] == tail[-NG_MIN - 1 - ext]):
                ext += 1
            if ext > best_ext:
                best_pos, best_ext = pos, ext
                if ext == NG_MAX - NG_MIN:
                    break
        return best_pos, best_ext

    def propose(self, out: list[int], budget: int) -> list[int]:
        """Copy block continuing ``prompt + out`` (empty when no match)."""
        if budget <= 0 or len(out) < self.suspend_until:
            return []
        n = NG_MAX + 1
        tail = out[-n:] if len(out) >= n else self.prompt[len(out) - n:] + out
        pos, ext = self._match(tail)
        if pos is None:
            return []
        cap = BLOCK_K if (self.seen >= 2 and self.ema >= 0.5) else PROBATION_K
        size = min(BLOCK_LADDER[min(ext, len(BLOCK_LADDER) - 1)], cap, budget)
        return self.prompt[pos:pos + size]

    def observe(self, out_len: int, block_len: int, accepted: int) -> None:
        ratio = accepted / block_len
        self.ema = 0.7 * self.ema + 0.3 * ratio
        self.seen += 1
        if ratio >= 0.5:
            self.backoff = 64
        if self.seen >= 3 and self.ema < 0.35:
            self.suspend_until = out_len + self.backoff
            self.backoff = min(self.backoff * 2, 4096)
            self.ema, self.seen = 0.5, 0
