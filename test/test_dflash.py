"""DFlash speculative-decoding integration tests.

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
    os.path.expanduser("~/.mtplx/models/Qwen3.6-27B-Q4-MTPLX-v2-Q2Mix11-L29UpQ4-Q3KO16"),
)
DRAFT = os.environ.get(
    "MULTIPLEX_TEST_DFLASH_DRAFT",
    os.path.join(TARGET, "dflash"),
)
HAS_MODELS = (
    os.path.isfile(os.path.join(TARGET, "config.json"))
    and os.path.isfile(os.path.join(DRAFT, "config.json"))
)


class _FakeEngine:
    def __init__(self, taps):
        self.last_hidden_taps = taps


class DFlashContextTests(unittest.TestCase):
    def test_ragged_commit_keeps_each_rows_context_length(self):
        import mlx.core as mx
        from multiplex.kernel.dflash import DFlashDrafter

        drafter = DFlashDrafter.__new__(DFlashDrafter)
        taps = mx.arange(2 * 6 * 3, dtype=mx.float32).reshape(2, 6, 3)
        ctx = drafter.update_context_after_commit(
            _FakeEngine(taps), None, [5, 1]
        )

        self.assertEqual([tuple(x.shape) for x in ctx], [(1, 6, 3), (1, 2, 3)])
        self.assertTrue(mx.all(ctx[0] == taps[0:1, :6, :]).item())
        self.assertTrue(mx.all(ctx[1] == taps[1:2, :2, :]).item())


class _FakeDrafter:
    """Minimal fixed-block drafter: pins its draft width but opts into adaptive
    trunk-verify (like DFlash)."""
    supports_dynamic_depth = False
    supports_adaptive_verify = True
    max_draft_len = 15

    def make_cache(self):
        return []


class DFlashAdaptiveVerifyWiringTests(unittest.TestCase):
    def _sched(self, drafter, **kw):
        from multiplex.kernel.scheduler import Scheduler
        return Scheduler(_FakeEngine(None), drafter, eos_token_ids=[0],
                         k=15, prefix_cache="0", **kw)

    def test_controller_built_for_adaptive_verify_even_with_dynamic_off(self):
        sch = self._sched(_FakeDrafter(), dynamic_depth=False)
        self.assertTrue(sch.adaptive_verify)
        self.assertIsNotNone(sch.depth_controller)
        # Draft width stays pinned at the fixed block; verify WARMS UP mid-block
        # (default 7) rather than at the full 15, so short generations benefit.
        self.assertEqual(sch.max_k, 15)
        self.assertEqual(sch.k, 7)

    def test_warm_start_clamps_to_max_k(self):
        sch = self._sched(_FakeDrafter(), dynamic_depth=False,
                          adaptive_verify_start=999)
        self.assertEqual(sch.k, sch.max_k)

    def test_reset_restarts_at_warm_start_not_full_block(self):
        sch = self._sched(_FakeDrafter(), dynamic_depth=False)
        sch.depth_controller.current = 3
        sch._reset_dynamic_depth(restart_at_max=True)
        self.assertEqual(sch.k, 7)

    def test_verify_width_narrows_on_low_acceptance(self):
        sch = self._sched(_FakeDrafter(), dynamic_depth=False)
        # Feed the controller a run of full-depth misses; verify width steps
        # down while max_k (the draft width) is untouched.
        for _ in range(16):
            sch.k = sch.depth_controller.observe(0).current
        self.assertLess(sch.k, sch.max_k)
        self.assertEqual(sch.max_k, 15)

    def test_plain_drafter_keeps_verify_equal_to_draft(self):
        class _Plain:
            supports_dynamic_depth = True
            supports_adaptive_verify = False
            max_draft_len = 3
            def make_cache(self):
                return []
        sch = self._sched(_Plain(), dynamic_depth=True)
        self.assertFalse(sch.adaptive_verify)


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
            # cache is a per-row list; each entry is that row's 5-layer draft
            # cache. This test runs B=1, so inspect row 0's full-attention layer.
            before = int(cache[0][-1].offset)
            ctx_len = 0 if ctx is None else int(ctx[0].shape[1])
            out = orig(self, ctx, primary, k, cache)
            deltas.append((ctx_len, int(cache[0][-1].offset) - before))
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
