"""Batch-lifecycle regression tests for the drafter contract.

The scheduler threads a per-row drafter context (``self.ctx``) alongside
``self.h``/``self.primary``. When a drafter draws from that context (as MTP and
DFlash both do), it MUST be re-sliced on leave (``_keep``) and rebuilt on join
(``merge_ready``) exactly like ``self.h`` — otherwise a batch-dimension mismatch
crashes the next ``draft()`` when a request joins or leaves mid-flight.

These tests reproduce a real regression: they fail with a shape mismatch in
``draft()`` if ``self.ctx`` is not kept row-aligned across join/leave.

They need a local MTP model; set ``MULTIPLEX_TEST_MTP_MODEL`` to override the
default path, or they skip.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multiplex.kernel.engine import Engine
from multiplex.kernel.mtp import find_drafter
from multiplex.kernel.scheduler import Scheduler, Req, PrefillGroup

MTP_MODEL = os.environ.get(
    "MULTIPLEX_TEST_MTP_MODEL",
    os.path.expanduser("~/.mtplx/models/Qwen3.5-2B-MTPLX-4bit-MTP4"),
)
HAS_MTP = os.path.isfile(os.path.join(MTP_MODEL, "config.json"))


@unittest.skipUnless(HAS_MTP, f"MTP model not found at {MTP_MODEL}")
class DrafterBatchLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = Engine(MTP_MODEL)
        cls.tok = cls.eng.tokenizer
        cls.drafter = find_drafter(cls.eng)
        assert cls.drafter is not None, "expected an MTP drafter for this model"

    def _fresh_scheduler(self):
        return Scheduler(
            self.eng, self.drafter,
            eos_token_ids=self.tok.eos_token_ids, k=3, debug=False,
        )

    def _ids(self, text):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True
        )

    def _admit(self, sch, rid, text, max_tokens):
        group = PrefillGroup(req=Req(rid, self._ids(text), max_tokens))
        while True:
            done = sch.prefill_chunk(group)
            self.assertIsNotNone(done, "prefill was cancelled unexpectedly")
            if done:
                break
        sch.merge_ready(group)

    def test_join_keeps_ctx_row_aligned(self):
        """A request joining a live batch (B=1 -> B=2) must not desync ctx."""
        sch = self._fresh_scheduler()
        self._admit(sch, 0, "Count slowly from one to fifty in words.", 200)
        for _ in range(4):
            sch.step()
        self.assertEqual(len(sch.rows), 1)

        self._admit(sch, 1, "Write a short poem about rain.", 200)
        self.assertEqual(len(sch.rows), 2)
        self.assertEqual(int(sch.h.shape[0]), 2)
        self.assertEqual(int(sch.ctx.shape[0]), 2)  # was 1 before the fix

        for _ in range(3):
            sch.step()  # would crash on the ctx/primary shape mismatch

    def test_leave_keeps_ctx_row_aligned(self):
        """A request finishing (B=2 -> B=1) must re-slice ctx with the survivors."""
        sch = self._fresh_scheduler()
        self._admit(sch, 0, "Write a long story about a dragon.", 300)
        self._admit(sch, 1, "Say hi.", 6)  # short -> finishes first -> _keep
        left = False
        for _ in range(40):
            sch.step()
            if len(sch.rows) == 1:
                left = True
                self.assertEqual(int(sch.h.shape[0]), 1)
                self.assertEqual(int(sch.ctx.shape[0]), 1)
                for _ in range(3):
                    sch.step()  # survivor keeps decoding
                break
        self.assertTrue(left, "expected the short request to finish and leave")


if __name__ == "__main__":
    unittest.main()
