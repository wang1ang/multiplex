"""Chat CLI for testing dynamic batching and mid-generation joins.

    python try_engine.py [--model PATH] [--raw] [-n N] [-d DEPTH] [--debug]
                         [--prompt TEXT | --prompt-file PATH]

A fixed input box sits at the bottom (like most CLIs); generated text scrolls
above it. Type a prompt + Enter to start; type another while it runs to add it
to the live batch. ``--prompt`` and ``--prompt-file`` submit an initial request
automatically. JSON/JSONL prompt files use the first object's ``prompt`` field.
:q or Ctrl-C quits.

Drives multiplex.scheduler.Scheduler: new requests are chunk-prefilled and
merged into the running batch. -d = fixed draft depth, or the maximum when
``--dynamic-depth`` is enabled (0 = pure AR).
"""

import argparse
import asyncio
import json
from pathlib import Path

import mlx.core as mx
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.document import Document

from multiplex import registry
from multiplex.engine import Engine
from multiplex.mtp import find_drafter
from multiplex.scheduler import Scheduler, Req, PrefillGroup


def to_ids(tokenizer, text, raw):
    if raw:
        return tokenizer.encode(text)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True
    )


def decode(tokenizer, token_ids, *, skip_special_tokens=True):
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def load_prompt_file(path: str) -> str:
    """Load plain text, or the first ``prompt`` field from JSON/JSONL."""
    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() not in {".json", ".jsonl"}:
        prompt = text.strip()
        if not prompt:
            raise ValueError(f"prompt file is empty: {source}")
        return prompt

    if source.suffix.lower() == ".jsonl":
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"prompt file is empty: {source}")
        payload = json.loads(lines[0])
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            if not payload:
                raise ValueError(f"prompt JSON array is empty: {source}")
            payload = payload[0]

    if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str):
        raise ValueError(f"prompt JSON must contain a string 'prompt' field: {source}")
    prompt = payload["prompt"]
    if not prompt.strip():
        raise ValueError(f"prompt field is empty: {source}")
    return prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="model path or name; default: scan ~/.mtplx/models")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("-n", "--max-tokens", type=int, default=8192)
    ap.add_argument("-d", "--depth", type=int, default=1)
    ap.add_argument(
        "--dynamic-depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="adapt D1..Dmax from live full-depth acceptance",
    )
    ap.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True,
                    help="show scheduler debug log in the log pane")
    initial = ap.add_mutually_exclusive_group()
    initial.add_argument("--prompt", help="submit this prompt when the UI starts")
    initial.add_argument(
        "--prompt-file",
        help="submit a text file, or the first prompt in a JSON/JSONL file",
    )
    args = ap.parse_args()
    initial_prompt = (
        load_prompt_file(args.prompt_file) if args.prompt_file else args.prompt
    )

    entry = registry.select(args.model)
    eng = Engine(entry.path)
    tokenizer = eng.tokenizer
    drafter = find_drafter(eng)
    print(f"[loaded {entry.name}{' + MTP head' if drafter else ' (headless, pure AR)'}]")

    debug_lines = []

    def append_debug(line):
        debug_lines.append(line)
        del debug_lines[:-80]

    sch = Scheduler(
        eng, drafter, eos_token_ids=tokenizer.eos_token_ids,
        k=args.depth, chunk=512, debug=args.debug,
        dynamic_depth=args.dynamic_depth,
        output_decode=lambda ids: decode(tokenizer, ids, skip_special_tokens=False),
        log=append_debug if args.debug else None,
    )

    prompts = {}         # rid -> prompt text
    produced_text = {}   # rid -> decoded output so far

    # Buffers are read-only panes; putting the cursor at the end makes each
    # window auto-scroll to the bottom (follow latest output).
    output_buf = Buffer(read_only=True)
    log_buf = Buffer(read_only=True)

    def render():
        output_lines = [
            "[Type a prompt + Enter. Add more while it runs. :q quits.]",
            "",
        ]
        for rid in sorted(produced_text):
            output_lines.append(f"--- req{rid}: {prompts.get(rid, '')[:50]!r}")
            output_lines.extend(produced_text[rid].split("\n"))
            output_lines.append("")
        text = "\n".join(output_lines)
        output_buf.set_document(
            Document(text, cursor_position=len(text)), bypass_readonly=True
        )

        log_lines = ["[scheduler log]", ""]
        log_lines.extend(debug_lines[-80:] if debug_lines else ["(no logs yet)"])
        log_text = "\n".join(log_lines)
        log_buf.set_document(
            Document(log_text, cursor_position=len(log_text)), bypass_readonly=True
        )

    # --- UI: output and log on top, fixed input box at bottom ---
    output_win = Window(content=BufferControl(buffer=output_buf), wrap_lines=True)
    log_win = Window(content=BufferControl(buffer=log_buf), wrap_lines=True)
    top = VSplit([output_win, Window(width=1, char="│"), log_win])
    input_buf = Buffer(multiline=False)
    input_win = Window(content=BufferControl(buffer=input_buf), height=1)
    layout = Layout(
        HSplit([top, Window(height=1, char="─"), input_win]),
        focused_element=input_win,
    )

    next_rid = [0]

    def add(text):
        rid = next_rid[0]
        next_rid[0] += 1
        prompts[rid] = text
        produced_text[rid] = ""
        # Prefill the new request and merge it into the live batch. The
        # merge returns each joined request's FIRST token — show it now (it is
        # not part of the next step()'s output).
        group = PrefillGroup(req=Req(rid, to_ids(tokenizer, text, args.raw), args.max_tokens))
        while True:
            done = sch.prefill_chunk(group)
            if done is None:
                return
            if done:
                break
        for r, first in sch.merge_ready(group):
            produced_text[r] += decode(tokenizer, [first])
        render()

    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        text = input_buf.text.strip()
        input_buf.reset()
        if text == ":q":
            event.app.exit()
        elif text:
            add(text)

    @kb.add("c-c")
    def _(event):
        event.app.exit()

    if initial_prompt:
        add(initial_prompt)
    else:
        render()

    app = Application(layout=layout, key_bindings=kb, full_screen=True,
                      mouse_support=True, refresh_interval=0.1)

    async def driver():
        # one scheduler step per loop iteration; yield to the UI between steps
        while True:
            if sch.has_rows():
                for rid, toks in sch.step():
                    produced_text[rid] = produced_text.get(rid, "") + decode(tokenizer, toks)
                render()
                app.invalidate()
            await asyncio.sleep(0.001)

    async def run_app():
        task = asyncio.create_task(driver())
        try:
            await app.run_async()
        finally:
            task.cancel()

    asyncio.run(run_app())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
