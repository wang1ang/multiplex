# SPDX-License-Identifier: Apache-2.0
"""Tool-call parsing and stream filtering adapted from oMLX.

Source inspiration: oMLX ``omlx/api/tool_calling.py`` on origin/main,
Apache-2.0. MTPLX intentionally uses this module as a protocol adapter, not as
a model babysitter: visible text is preserved, raw control markup is filtered,
and valid tool calls are parsed at completion.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

_INVOKE = r'<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>'
_PARAM_EQ = r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>"
_PARAM_NAMED = r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>'
_NAMESPACED = r"<([A-Za-z_][\w.-]*):tool_call>\s*(.*?)\s*</\1:tool_call>"
_NAMESPACED_OPEN = re.compile(r"<([A-Za-z_][\w.-]*):tool_call>")
_BRACKET = r"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)(?:\(({.*?})\))?\]"
_BRACKET_PREFIXES = ("[Calling tool:", "[Tool call:")
_OPEN, _CLOSE = "<tool_call>", "</tool_call>"


def _serialize_tool_call_arguments(arguments: Any) -> str:
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return "{}"


_TOOL_ARGUMENT_PRIMARY_ORDER: dict[str, tuple[str, ...]] = {
    "read": ("filePath", "path", "offset", "limit"),
    "grep": ("pattern", "path", "include", "limit"),
    "glob": ("pattern", "path"),
    "bash": ("command", "description", "timeout"),
    "write": ("filePath", "path", "content"),
    "edit": ("filePath", "path", "oldString", "newString", "replaceAll"),
    "webfetch": ("url", "format", "timeout"),
    "skill": ("name",),
    "question": ("questions",),
}


def _order_tool_arguments_for_client_display(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Return stable argument order so clients do not lead with display noise.

    OpenCode renders tool argument values in JSON insertion order. Qwen can emit
    optional knobs such as ``limit`` before the useful target path, producing
    transcript rows that start with a bare ``100``. This preserves every
    argument and only moves the user-facing identifiers first.
    """

    if not arguments:
        return arguments
    order = _TOOL_ARGUMENT_PRIMARY_ORDER.get(str(tool_name or "").strip().lower())
    if not order:
        return arguments
    ordered: dict[str, Any] = {}
    for key in order:
        if key in arguments:
            ordered[key] = arguments[key]
    for key, value in arguments.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _tool_call(name: str, arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        arguments = _order_tool_arguments_for_client_display(str(name), arguments)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": _serialize_tool_call_arguments(arguments),
        },
    }


def _coerce_json_tool_payload(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, list):
        for item in parsed:
            coerced = _coerce_json_tool_payload(item)
            if coerced is not None:
                return coerced
        return None
    if not isinstance(parsed, dict):
        return None
    function = parsed.get("function")
    if isinstance(function, dict):
        name = function.get("name") or function.get("tool") or function.get("function")
        arguments = (
            function.get("arguments")
            if "arguments" in function
            else function.get("args", function.get("parameters", {}))
        )
        if name:
            return _tool_call(str(name), arguments)
    name = (
        parsed.get("name")
        or parsed.get("tool")
        or parsed.get("function")
        or parsed.get("call")
    )
    if not name:
        return None
    arguments = parsed.get("arguments", parsed.get("args", parsed.get("parameters", {})))
    return _tool_call(str(name), arguments)


def _parse_json_tool_payload(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    return _coerce_json_tool_payload(parsed)


def _parse_bare_tool_payload(content: str) -> dict[str, Any] | None:
    stripped = content.strip()
    match = re.match(
        r"^([A-Za-z_][\w.-]*)\s*(?:\((.*)\)|:\s*(\{.*\})|(\{.*\}))\s*$",
        stripped,
        re.DOTALL,
    )
    if not match:
        return None
    name = match.group(1)
    raw_args = next((group for group in match.groups()[1:] if group), "{}")
    try:
        arguments = json.loads(raw_args)
    except (TypeError, ValueError):
        arguments = {"_raw": raw_args}
    return _tool_call(name, arguments)


def _xml_params(body: str, patterns: tuple[str, ...]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for pattern in patterns:
        for param in re.finditer(pattern, body, re.DOTALL):
            value = param.group(2).strip()
            try:
                params[param.group(1)] = json.loads(value)
            except (TypeError, ValueError):
                params[param.group(1)] = value
    return params


def _parse_xml_tool_calls(text: str) -> list[dict[str, Any]] | None:
    calls: list[dict[str, Any]] = []
    for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        content = match.strip()
        if parsed := _parse_json_tool_payload(content):
            calls.append(parsed)
            continue
        if parsed := _parse_bare_tool_payload(content):
            calls.append(parsed)
            continue
        func_match = re.match(
            r"<function=([^>\s]+)>\s*(.*?)\s*</function>",
            content,
            re.DOTALL,
        )
        if func_match is None:
            func_match = re.match(
                r'<function\s+name="([^"]+)">\s*(.*?)\s*</function>',
                content,
                re.DOTALL,
            )
        if func_match:
            params = _xml_params(func_match.group(2), (_PARAM_EQ, _PARAM_NAMED))
            calls.append(_tool_call(func_match.group(1), params))
            continue
        if invoke_match := re.match(_INVOKE, content, re.DOTALL):
            params = _xml_params(invoke_match.group(2), (_PARAM_NAMED,))
            calls.append(_tool_call(invoke_match.group(1), params))
    return calls or None


def _parse_namespaced_tool_calls(text: str) -> list[dict[str, Any]] | None:
    calls = [
        _tool_call(invoke.group(1), _xml_params(invoke.group(2), (_PARAM_NAMED,)))
        for _namespace, content in re.findall(_NAMESPACED, text, re.DOTALL)
        for invoke in re.finditer(_INVOKE, content, re.DOTALL)
    ]
    return calls or None


def _parse_bracket_tool_calls(text: str) -> list[dict[str, Any]] | None:
    calls: list[dict[str, Any]] = []
    for name, args_text in re.findall(_BRACKET, text, re.DOTALL):
        arguments: Any = {}
        if args_text:
            try:
                arguments = json.loads(args_text)
            except (TypeError, ValueError):
                arguments = {}
        calls.append(_tool_call(name, arguments))
    return calls or None


def _allowed_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
        elif isinstance(tool, dict) and tool.get("name"):
            names.add(str(tool["name"]))
    return names


def _filter_known_tools(
    calls: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not calls or not tools:
        return calls
    allowed = _allowed_tool_names(tools)
    filtered = [
        call
        for call in calls
        if call.get("function", {}).get("name") in allowed
    ]
    return filtered or None


def parse_tool_calls(
    text: str,
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]] | None:
    """Return the tool calls in a finished completion, or None. Calls to tools
    not offered in ``tools`` are dropped."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    if "<tool_call" in text:
        # Qwen XML is authoritative: when it has no valid call, the text stays
        # content rather than falling through to the looser formats.
        return _filter_known_tools(_parse_xml_tool_calls(text), tools)
    if _NAMESPACED_OPEN.search(text):
        if calls := _filter_known_tools(_parse_namespaced_tool_calls(text), tools):
            return calls
    if any(prefix in text for prefix in _BRACKET_PREFIXES):
        if calls := _filter_known_tools(_parse_bracket_tool_calls(text), tools):
            return calls
    return None


class ToolCallStreamFilter:
    """Suppress tool-control markup while preserving normal streamed text."""

    def __init__(self) -> None:
        self._bracket_call_re = re.compile("^" + _BRACKET, re.DOTALL)
        self._buffer = ""
        self._suppressing_until: str | None = None

    def _find_start_envelope(self, text: str) -> tuple[int, int, str | None] | None:
        starts: list[tuple[int, int, str | None]] = []
        if (index := text.find(_OPEN)) >= 0:
            starts.append((index, len(_OPEN), _CLOSE))
        if match := _NAMESPACED_OPEN.search(text):
            namespace = match.group(1)
            starts.append((match.start(), len(match.group(0)), f"</{namespace}:tool_call>"))
        for prefix in _BRACKET_PREFIXES:
            index = text.find(prefix)
            while index >= 0:
                if bracket := self._bracket_call_re.match(text[index:]):
                    starts.append((index, bracket.end(), None))
                index = text.find(prefix, index + 1)
        return min(starts, key=lambda item: item[0]) if starts else None

    @staticmethod
    def _partial_prefix_len(text: str, marker: str) -> int:
        max_len = min(len(text), len(marker) - 1)
        for size in range(max_len, 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0

    @staticmethod
    def _could_be_partial_namespaced_open(candidate: str) -> bool:
        if not candidate.startswith("<") or ">" in candidate:
            return False
        body = candidate[1:]
        if not body or body.startswith("/"):
            return bool(not body)
        if ":" not in body:
            return re.match(r"^[A-Za-z_][\w.-]*$", body) is not None
        namespace, suffix = body.split(":", 1)
        return bool(
            re.match(r"^[A-Za-z_][\w.-]*$", namespace)
            and "tool_call".startswith(suffix)
        )

    def _partial_suffix_len(self, text: str) -> int:
        keep = self._partial_prefix_len(text, _OPEN)
        if (last_lt := text.rfind("<")) >= 0:
            candidate = text[last_lt:]
            if self._could_be_partial_namespaced_open(candidate):
                keep = max(keep, len(candidate))
        for prefix in _BRACKET_PREFIXES:
            keep = max(keep, self._partial_prefix_len(text, prefix))
            if (index := text.rfind(prefix)) >= 0 and "]" not in text[index:]:
                return max(keep, len(text[index:]))
        return min(keep, 128)

    @staticmethod
    def _should_drop_tail_at_finish(tail: str) -> bool:
        if not tail:
            return False
        if _OPEN.startswith(tail) or tail.startswith(_BRACKET_PREFIXES):
            return True
        return tail.startswith("<") and ">" not in tail and ":" in tail

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buffer += text
        out: list[str] = []
        while self._buffer:
            if self._suppressing_until is not None:
                end_index = self._buffer.find(self._suppressing_until)
                if end_index < 0:
                    keep = self._partial_prefix_len(self._buffer, self._suppressing_until)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._buffer = self._buffer[end_index + len(self._suppressing_until) :]
                self._suppressing_until = None
                continue
            start = self._find_start_envelope(self._buffer)
            if start:
                index, consume_len, close_marker = start
                if index > 0:
                    out.append(self._buffer[:index])
                self._buffer = self._buffer[index + consume_len :]
                if close_marker is not None:
                    self._suppressing_until = close_marker
                continue
            keep = self._partial_suffix_len(self._buffer)
            if keep == 0:
                out.append(self._buffer)
                self._buffer = ""
                break
            if len(self._buffer) > keep:
                out.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
            break
        return "".join(out)

    def finish(self) -> str:
        if self._suppressing_until is not None:
            self._buffer = ""
            self._suppressing_until = None
            return ""
        keep = self._partial_suffix_len(self._buffer)
        if keep >= len(self._buffer):
            tail = self._buffer
            self._buffer = ""
            return "" if self._should_drop_tail_at_finish(tail) else tail
        text = self._buffer[:-keep] if keep else self._buffer
        self._buffer = ""
        return text
