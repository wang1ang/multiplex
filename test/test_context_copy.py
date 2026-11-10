from multiplex.kernel.context_copy import BLOCK_K, PROBATION_K, ContextCopy


def test_proposes_prompt_continuation_after_ngram_match():
    cc = ContextCopy(list(range(100)))
    # output re-emits prompt[10:20]; continuation is prompt[20:]
    block = cc.propose(list(range(10, 20)), budget=1000)
    assert block == list(range(20, 20 + PROBATION_K))     # probation cap


def test_match_spans_prompt_and_output_boundary():
    prompt = [7, 7, 7] + list(range(10, 30)) + [1, 2, 3, 4, 5, 6]
    cc = ContextCopy(prompt)
    # Empty output: the history tail is the prompt's own last gram, which has
    # nothing after it to copy -> no match. A short output is joined with the
    # prompt's tail to form the lookup key.
    assert cc.propose([], budget=10) == []
    assert cc.propose([10, 11, 12, 13, 14, 15], budget=10)[:3] == [16, 17, 18]


def test_no_match_and_budget():
    cc = ContextCopy(list(range(100)))
    assert cc.propose([500, 501, 502, 503, 504, 505], budget=1000) == []
    assert len(cc.propose(list(range(10, 20)), budget=3)) == 3
    assert cc.propose(list(range(10, 20)), budget=0) == []


def test_proven_lane_opens_full_block():
    cc = ContextCopy(list(range(200)))
    for _ in range(2):
        cc.observe(10, 8, 8)
    # 20-token history -> full backward extension (4) -> ladder 32, capped at 24
    assert len(cc.propose(list(range(0, 20)), budget=1000)) == BLOCK_K


def test_suspends_on_low_acceptance_with_doubling_backoff():
    cc = ContextCopy(list(range(100)))
    for _ in range(3):
        cc.observe(50, 8, 0)
    assert cc.suspend_until == 50 + 64 and cc.backoff == 128
    assert cc.propose(list(range(10, 20)), budget=1000) == []   # len(out)=10 < 114
