import json

import mlx.core as mx

from multiplex.kernel.hub import Hub
from multiplex.kernel.scheduler import Req, Scheduler
from multiplex.server import (
    _chat_stream,
    _collect_parts,
    _enable_thinking_from_body,
    _responses_body,
    _responses_stream,
)


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


class TinyTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt=True, **kwargs):
        body = "\n".join(
            f"{item['role']}:{item.get('content') or ''}" for item in messages
        )
        if add_generation_prompt:
            body += "\nassistant:"
            if kwargs.get("enable_thinking") is False:
                body += "<think>\n\n</think>\n\n"
            else:
                body += "<think>\n"
        return _ids(body)

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token)) for token in token_ids)


def _hub_with_raw_chunks(chunks):
    hub = Hub.__new__(Hub)
    hub.tokenizer = TinyTokenizer()
    hub._default_enable_thinking = None
    hub._stream_prompt_ids = lambda _prompt_ids, _max_tokens, **_kw: iter(chunks)
    return hub


def _hub_recording_decode_params():
    calls = []
    hub = _hub_with_raw_chunks(["ok"])

    def stream(_prompt_ids, _max_tokens, *, outcome, **kwargs):
        calls.append(kwargs)
        return iter(["ok"])

    hub._stream_prompt_ids = stream
    return hub, calls


def test_l4_splits_prefilled_qwen_thinking_before_l5():
    hub = _hub_with_raw_chunks(["private ", "plan</thi", "nk>Visible answer"])

    parts = list(
        hub.stream_message_parts(
            [{"role": "user", "content": "hi"}],
            32,
            enable_thinking=True,
        )
    )

    reasoning = "".join(text for field, text in parts if field == "reasoning_content")
    content = "".join(text for field, text in parts if field == "content")

    assert reasoning == "private plan"
    assert content == "Visible answer"
    assert "</think>" not in content


def test_l4_reasoning_off_strips_orphan_close_marker():
    hub = _hub_with_raw_chunks(["</think>Visible answer"])

    parts = list(
        hub.stream_message_parts(
            [{"role": "user", "content": "hi"}],
            32,
            enable_thinking=False,
        )
    )

    assert parts == [("content", "Visible answer"), ("finish_reason", "stop")]


def test_l5_chat_stream_maps_l4_reasoning_to_reasoning_content_delta():
    class Backend:
        model_id = "tiny"

        def stream_message_parts(self, *_args, **_kwargs):
            yield "reasoning_content", "private plan"
            yield "content", "Visible answer"

    payloads = []
    for event in _chat_stream(
        Backend(), [{"role": "user", "content": "hi"}], 32
    ):
        if not event.startswith("data: ") or event.strip() == "data: [DONE]":
            continue
        payloads.append(json.loads(event[len("data: ") :]))

    deltas = [payload["choices"][0]["delta"] for payload in payloads]
    assert {"reasoning_content": "private plan"} in deltas
    assert {"content": "Visible answer"} in deltas


def test_l5_maps_reasoning_effort_to_enable_thinking():
    assert _enable_thinking_from_body({"reasoning": {"effort": "low"}}) is False
    assert _enable_thinking_from_body({"reasoning": {"effort": "minimal"}}) is False
    assert _enable_thinking_from_body({"reasoning": {"effort": "medium"}}) is True
    assert _enable_thinking_from_body({"reasoning": {"effort": "high"}}) is True
    assert _enable_thinking_from_body({"reasoning": {"effort": "xhigh"}}) is True
    assert _enable_thinking_from_body({"reasoning_effort": "high"}) is True


def test_l5_enable_thinking_overrides_reasoning_effort():
    assert _enable_thinking_from_body({
        "enable_thinking": False,
        "reasoning": {"effort": "high"},
    }) is False
    assert _enable_thinking_from_body({
        "enable_thinking": True,
        "reasoning": {"effort": "low"},
    }) is True


def test_l4_sets_decode_params_from_thinking_state():
    cases = [
        ({"enable_thinking": True}, {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
        ({}, {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
        ({"enable_thinking": False}, {"temperature": 0.7, "top_p": 0.8, "top_k": 20}),
    ]
    for kwargs, expected in cases:
        hub, calls = _hub_recording_decode_params()
        list(hub.stream_message_parts(
            [{"role": "user", "content": "hi"}], 8, **kwargs
        ))
        assert calls == [expected]


def test_l3_accepts_draft_by_probability(monkeypatch):
    req = Req(0, [], 1, temperature=1.0)
    logits = mx.array([2.0, 1.0, 0.0])
    scheduler = Scheduler.__new__(Scheduler)

    monkeypatch.setattr("multiplex.kernel.scheduler.random.random", lambda: 0.3)
    assert scheduler._accept_draft(req, logits, best=0, draft=1) is True

    monkeypatch.setattr("multiplex.kernel.scheduler.random.random", lambda: 0.5)
    assert scheduler._accept_draft(req, logits, best=0, draft=1) is False


def test_l3_accept_draft_top_k_and_top_p_are_hard_filters(monkeypatch):
    logits = mx.array([3.0, 2.0, 1.0])
    scheduler = Scheduler.__new__(Scheduler)
    monkeypatch.setattr("multiplex.kernel.scheduler.random.random", lambda: 0.0)

    assert scheduler._accept_draft(
        Req(0, [], 1, temperature=1.0, top_k=1), logits, best=0, draft=1
    ) is False
    assert scheduler._accept_draft(
        Req(0, [], 1, temperature=1.0, top_p=0.5), logits, best=0, draft=1
    ) is False


def _hub_truncated(chunks):
    """A hub whose generation hits max_tokens after ``chunks``."""
    hub = _hub_with_raw_chunks(chunks)

    def stream(_prompt_ids, _max_tokens, *, outcome, **_kw):
        yield from chunks
        outcome["finish_reason"] = "length"

    hub._stream_prompt_ids = stream
    return hub


def test_l4_truncated_thinking_stays_reasoning_and_reports_length():
    hub = _hub_truncated(["still ", "planning"])

    parts = list(hub.stream_message_parts(
        [{"role": "user", "content": "hi"}], 8, enable_thinking=True,
    ))

    reasoning = "".join(t for f, t in parts if f == "reasoning_content")
    assert reasoning == "still planning"
    assert [t for f, t in parts if f == "content"] == []
    assert parts[-1] == ("finish_reason", "length")


class _TruncatedBackend:
    model_id = "tiny"

    def __init__(self):
        self.kwargs = None

    def stream_message_parts(self, *_args, **kwargs):
        self.kwargs = kwargs
        yield "reasoning_content", "plan"
        yield "content", "partial"
        yield "finish_reason", "length"


def test_l5_reports_length():
    backend = _TruncatedBackend()
    content, reasoning, calls, finish = _collect_parts(
        backend, [{"role": "user", "content": "hi"}], 8, None,
    )
    assert (content, reasoning, calls, finish) == ("partial", "plan", [], "length")

    payloads = [json.loads(e[len("data: "):]) for e in _chat_stream(
        _TruncatedBackend(), [{"role": "user", "content": "hi"}], 8)
        if e.startswith("data: ") and e.strip() != "data: [DONE]"]
    assert payloads[-1]["choices"][0]["finish_reason"] == "length"
    contents = [p["choices"][0]["delta"].get("content") for p in payloads]
    assert contents.count("partial") == 1


def test_l5_responses_marks_truncation_incomplete():
    body = _responses_body(_TruncatedBackend(), "partial", None, "length")
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert _responses_body(_TruncatedBackend(), "done")["status"] == "completed"

    events = "".join(_responses_stream(
        _TruncatedBackend(), [{"role": "user", "content": "hi"}], 8))
    assert "response.incomplete" in events
    assert "response.completed" not in events


def test_scheduler_emit_records_finish_reason():
    sched = Scheduler.__new__(Scheduler)
    sched.eos = {0}
    sched._log_output = lambda *_a: None

    req = Req(rid=1, prompt=[5], max_tokens=10)
    assert sched._emit(req, [7, 0, 8]) == ([7, 0], True)
    assert req.finish_reason == "stop"

    req = Req(rid=2, prompt=[5], max_tokens=3)
    req.out = [7]
    assert sched._emit(req, [8, 9, 10]) == ([8, 9], True)
    assert req.finish_reason == "length"
