# SPDX-License-Identifier: Apache-2.0
"""Thinking parser adapted from oMLX.

Source inspiration: oMLX ``omlx/api/thinking.py`` on origin/main, Apache-2.0.
An unclosed thinking block stays reasoning. Re-sending it as content (oMLX's
recovery) duplicates everything a streaming client already received; a
generation cut off while thinking is reported as finish_reason "length".
"""

from __future__ import annotations

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_OPEN_LEN = len(THINK_OPEN)
_CLOSE_LEN = len(THINK_CLOSE)


class ThinkingParser:
    """Streaming ``<think>`` splitter: text inside the block is reasoning."""

    def __init__(self, *, starts_in_thinking: bool = False) -> None:
        self._in_thinking = starts_in_thinking
        self._buffer = ""

    def feed(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        text = self._buffer + text
        self._buffer = ""
        thinking_out: list[str] = []
        content_out: list[str] = []
        i = 0
        while i < len(text):
            if text[i] == "<":
                remaining = text[i:]
                if remaining.startswith(THINK_OPEN):
                    self._in_thinking = True
                    i += _OPEN_LEN
                    continue
                if remaining.startswith(THINK_CLOSE):
                    self._in_thinking = False
                    i += _CLOSE_LEN
                    continue
                if self._could_be_tag(remaining):
                    self._buffer = remaining
                    break
            if self._in_thinking:
                thinking_out.append(text[i])
            else:
                content_out.append(text[i])
            i += 1
        return "".join(thinking_out), "".join(content_out)

    def finish(self) -> tuple[str, str]:
        partial, self._buffer = self._buffer, ""
        return (partial, "") if self._in_thinking else ("", partial)

    @staticmethod
    def _could_be_tag(text: str) -> bool:
        if len(text) >= _CLOSE_LEN:
            return False
        return THINK_OPEN.startswith(text) or THINK_CLOSE.startswith(text)
