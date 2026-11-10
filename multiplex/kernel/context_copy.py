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
    longer block (_BLOCK_LADDER), so a weak match only risks a short verify;
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
_BLOCK_LADDER = (8, 12, 16, 24, 32)


def block_for_ext(ext: int, k_cap: int) -> int:
    idx = max(0, min(int(ext), len(_BLOCK_LADDER) - 1))
    return min(_BLOCK_LADDER[idx], max(4, k_cap))


class NgramIndex:
    """NG_MIN-gram -> continuation positions over the prompt."""

    def __init__(self, prompt: list[int], ng_min: int = NG_MIN, ng_max: int = NG_MAX):
        self.ng_min = ng_min
        self.ng_max = ng_max
        self.grams: dict[tuple, list[int]] = {}
        for e in range(ng_min, len(prompt) + 1):
            self.grams.setdefault(tuple(prompt[e - ng_min:e]), []).append(e)

    def find(self, history, max_pos: int):
        """Best match ``(continuation_pos, extension)`` or ``(None, -1)``.

        ``history`` is prompt + output; ``max_pos`` (exclusive) keeps only
        matches with a prompt continuation left to copy. Extension = how many
        tokens beyond ng_min the match runs backwards; longest wins, most
        recent breaks ties.
        """
        L = len(history)
        if L < self.ng_min + 1:
            return None, -1
        cands = self.grams.get(tuple(history[L - self.ng_min:L]))
        if not cands:
            return None, -1
        best_pos, best_ext = None, -1
        max_ext = self.ng_max - self.ng_min
        for pos in reversed(cands[-MAX_CANDIDATES:]):
            if pos >= L or pos >= max_pos:
                continue
            ext = 0
            while (ext < max_ext
                   and pos - self.ng_min - 1 - ext >= 0
                   and history[pos - self.ng_min - 1 - ext]
                   == history[L - self.ng_min - 1 - ext]):
                ext += 1
            if ext > best_ext:
                best_ext, best_pos = ext, pos
                if ext == max_ext:
                    break
        return best_pos, best_ext


class _Joined:
    """prompt + out as one read-only sequence, without copying the prompt."""

    __slots__ = ("head", "tail", "split")

    def __init__(self, head: list[int], tail: list[int]):
        self.head, self.tail, self.split = head, tail, len(head)

    def __len__(self):
        return self.split + len(self.tail)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return self.head[i] if i < self.split else self.tail[i - self.split]


class ContextCopy:
    """Per-request proposer + acceptance policy."""

    def __init__(self, prompt: list[int]):
        self.prompt = prompt
        self.index = NgramIndex(prompt)
        self.ema, self.seen = 0.5, 0
        self.suspend_until, self.backoff = 0, 64
        self.rounds = self.drafted = self.accepted = self.suspensions = 0

    def propose(self, out: list[int], budget: int) -> list[int]:
        """Copy block continuing ``prompt + out`` (empty when no match)."""
        if budget <= 0 or len(out) < self.suspend_until:
            return []
        pos, ext = self.index.find(_Joined(self.prompt, out), max_pos=len(self.prompt))
        if pos is None:
            return []
        cap = BLOCK_K if (self.seen >= 2 and self.ema >= 0.5) else PROBATION_K
        n = min(block_for_ext(ext, cap), budget)
        return self.prompt[pos:pos + n]

    def observe(self, out_len: int, block_len: int, accepted: int) -> None:
        ratio = accepted / block_len
        self.rounds += 1
        self.drafted += block_len
        self.accepted += accepted
        self.ema = 0.7 * self.ema + 0.3 * ratio
        self.seen += 1
        if ratio >= 0.5:
            self.backoff = 64
        if self.seen >= 3 and self.ema < 0.35:
            self.suspend_until = out_len + self.backoff
            self.backoff = min(self.backoff * 2, 4096)
            self.ema, self.seen = 0.5, 0
            self.suspensions += 1
