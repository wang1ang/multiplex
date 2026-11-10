from multiplex.kernel.hub import _DONE, RequestManager

# Byte-level fake tokenizer: each token is one UTF-8 byte, decoded like a real
# BPE tokenizer would (incomplete sequences become U+FFFD).
def _decode(ids):
    return bytes(ids).decode("utf-8", errors="replace")


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def test_character_split_across_tokens_is_sent_whole():
    rm = RequestManager(_decode)
    req, q = rm.submit([], 16)
    emoji = list("😊".encode())
    rm.emit([(req.rid, list(b"Hi! "))])
    for b in emoji:
        rm.emit([(req.rid, [b])])
    rm.finish(req.rid, notify=True)
    parts = _drain(q)
    assert parts[-1] is _DONE
    assert "".join(parts[:-1]) == "Hi! 😊"


def test_invalid_trailing_bytes_are_flushed_on_finish():
    rm = RequestManager(_decode)
    req, q = rm.submit([], 16)
    rm.emit([(req.rid, list(b"ok") + [0xF0, 0x9F])])
    rm.finish(req.rid, notify=True)
    parts = _drain(q)
    assert "".join(parts[:-1]) == "ok�"
