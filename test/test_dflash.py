"""DFlash speculative-decoding integration tests (single-stream, v1).

Exercises the DFlash product path through the real scheduler:
  * end-to-end generation via the Hub produces coherent tokens;
  * the draft cache is only ever fed committed context — its offset advances by
    exactly the committed length each step, so no trim/rollback is needed
    (a regression guard: if the drafted block ever leaked into the draft cache,
    the offset would jump by more than the fed context).

Needs a local Qwen3.6-27B target and the DFlash draft model. Override with
``MULTIPLEX_TEST_DFLASH_TARGET`` / ``MULTIPLEX_TEST_DFLASH_DRAFT``; otherwise the
tests skip.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGET = os.environ.get(
    "MULTIPLEX_TEST_DFLASH_TARGET",
    os.path.expanduser("~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"),
)
DRAFT = os.environ.get(
    "MULTIPLEX_TEST_DFLASH_DRAFT",
    os.path.expanduser("~/models/Qwen3.6-27B-DFlash"),
)
HAS_MODELS = (
    os.path.isfile(os.path.join(TARGET, "config.json"))
    and os.path.isfile(os.path.join(DRAFT, "config.json"))
)


@unittest.skipUnless(HAS_MODELS, f"DFlash target/draft not found ({TARGET}; {DRAFT})")
class DFlashSchedulerTests(unittest.TestCase):
    def test_end_to_end_generates(self):
        from multiplex.kernel.hub import Hub

        hub = Hub(TARGET, None, k=3, debug=False, dflash_path=DRAFT, prefix_cache="0")
        msgs = [{"role": "user", "content": "Write a quicksort in Python."}]
        text = "".join(
            t for field, t in hub.stream_message_parts(msgs, max_tokens=48)
            if field == "content"
        )
        self.assertGreater(len(text.strip()), 0)

    def test_draft_cache_only_holds_committed_context(self):
        """Each draft feeds exactly the committed positions' taps; the draft
        cache offset must advance by that many and no more (the block is never
        cached, so nothing needs trimming)."""
        import mlx.core as mx
        from multiplex.kernel import dflash as D
        from multiplex.kernel.hub import Hub

        deltas = []
        orig = D.DFlashDrafter.draft

        def traced(self, ctx, primary, k, cache):
            before = int(cache[-1].offset)
            ctx_len = 0 if ctx is None else int(ctx.shape[1])
            out = orig(self, ctx, primary, k, cache)
            deltas.append((ctx_len, int(cache[-1].offset) - before))
            return out

        D.DFlashDrafter.draft = traced
        try:
            hub = Hub(TARGET, None, k=3, debug=False, dflash_path=DRAFT, prefix_cache="0")
            msgs = [{"role": "user", "content": "Count from 1 to 20 in words."}]
            for _field, _t in hub.stream_message_parts(msgs, max_tokens=64):
                pass
        finally:
            D.DFlashDrafter.draft = orig

        self.assertTrue(deltas, "no draft steps ran")
        for ctx_len, delta in deltas:
            self.assertEqual(delta, ctx_len,
                             "draft cache advanced by != fed context "
                             "(block leaked into the draft cache?)")


if __name__ == "__main__":
    unittest.main()
