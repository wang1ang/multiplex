from multiplex.kernel.context_copy import BLOCK_K, PROBATION_K, ContextCopy, block_for_ext


def test_proposes_prompt_continuation_after_ngram_match():
    prompt = list(range(100))
    cc = ContextCopy(prompt)
    # output re-emits prompt[10:20]; continuation is prompt[20:]
    block = cc.propose(list(range(10, 20)), budget=1000)
    assert block == list(range(20, 20 + len(block)))
    assert 0 < len(block) <= PROBATION_K          # probation until proven


def test_no_match_and_budget():
    cc = ContextCopy(list(range(100)))
    assert cc.propose([500, 501, 502, 503, 504, 505], budget=1000) == []
    assert len(cc.propose(list(range(10, 20)), budget=3)) == 3
    assert cc.propose(list(range(10, 20)), budget=0) == []


def test_proven_lane_opens_full_block():
    cc = ContextCopy(list(range(200)))
    for _ in range(2):
        cc.observe(10, 8, 8)
    block = cc.propose(list(range(0, 20)), budget=1000)
    assert len(block) == block_for_ext(4, BLOCK_K) == 24


def test_suspends_on_low_acceptance_with_doubling_backoff():
    cc = ContextCopy(list(range(100)))
    for _ in range(3):
        cc.observe(50, 8, 0)
    assert cc.suspensions == 1 and cc.suspend_until == 50 + 64
    assert cc.propose(list(range(10, 20)), budget=1000) == []   # len(out)=10 < 114
    assert cc.backoff == 128
