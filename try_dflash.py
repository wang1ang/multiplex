"""DFlash speculative decoding — dual mode.

No arguments -> interactive chat CLI, like ``try_engine.py`` but on the DFlash
product path (Engine + build_dflash_drafter + hidden taps + Scheduler). Picks the
target from ``~/.mtplx/models`` and defaults the draft to
``~/models/Qwen3.6-27B-DFlash``.

    python try_dflash.py

With arguments -> the original one-shot reference smoke test: run the vendored
DFlash MLX ``stream_generate`` (single-sequence, matching the official reference)
and print acceptance / throughput stats.

    python try_dflash.py --target <dir> --draft <dir> [--prompt ...] [--max-tokens N]

DFlash is batched: prompts typed while a response is running join the live
decode batch (like ``try_engine.py``). :q or Ctrl-C quits the interactive UI.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from multiplex import registry
from multiplex.kernel.engine import Engine
from multiplex.kernel.dflash import (
    build_dflash_drafter, load, load_draft, stream_generate,
)
from multiplex.kernel.scheduler import Scheduler, Req, PrefillGroup

DEFAULT_DRAFT = "~/models/Qwen3.6-27B-DFlash"


def to_ids(tokenizer, text, raw, think=None):
    if raw:
        return tokenizer.encode(text)
    kwargs = {}
    if think is not None:
        kwargs["enable_thinking"] = think
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, **kwargs
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


# ---------------------------------------------------------------------------
# reference mode (args): one-shot stream_generate smoke test
# ---------------------------------------------------------------------------

def main_reference(argv) -> int:
    ap = argparse.ArgumentParser(prog="try_dflash.py (reference mode)")
    ap.add_argument("--target", required=True, help="target model dir or HF id")
    ap.add_argument("--draft", default=DEFAULT_DRAFT, help="DFlash draft dir or HF id")
    ap.add_argument("--prompt", default="Write a quicksort in Python.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--block-size", type=int, default=None)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    model, tokenizer = load(args.target)
    draft = load_draft(os.path.expanduser(args.draft))
    print(f"[load] target+draft in {time.perf_counter() - t0:.1f}s")

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    accepted_lens, n_blocks = [], 0
    final = None
    print("-" * 60)
    for resp in stream_generate(
        model, draft, tokenizer, prompt,
        block_size=args.block_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ):
        print(resp.text, end="", flush=True)
        final = resp
        if resp.accepted:
            accepted_lens.append(int(resp.accepted))
            n_blocks += 1
    print("\n" + "-" * 60)
    if final is not None:
        mean_acc = sum(accepted_lens) / len(accepted_lens) if accepted_lens else 0.0
        print(f"[done] {final.generation_tokens} tok @ {final.generation_tps:.1f} tok/s | "
              f"blocks={n_blocks} mean_accept={mean_acc:.2f} | "
              f"peak={final.peak_memory:.1f}GB | finish={final.finish_reason}")
    return 0


# ---------------------------------------------------------------------------
# interactive mode (no args): try_engine-style chat CLI on the DFlash path
# ---------------------------------------------------------------------------

def main_interactive() -> int:
    # prompt_toolkit is a CLI-only dep (pip install -e ".[cli]"); import it lazily
    # so the reference mode above works without it.
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
    from prompt_toolkit.layout.controls import BufferControl
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.document import Document

    draft_dir = os.path.expanduser(DEFAULT_DRAFT)
    if not os.path.isdir(draft_dir):
        print(f"[error] default DFlash draft dir not found: {draft_dir}\n"
              f"        run with args for reference mode, or place the draft there.",
              file=sys.stderr)
        return 1

    entry = registry.select(None)
    eng = Engine(entry.path)
    tokenizer = eng.tokenizer
    drafter = build_dflash_drafter(eng, draft_dir)
    eng.enable_hidden_taps(drafter.tap_layer_ids)
    print(f"[loaded {entry.name} + DFlash drafter "
          f"(block={drafter.block_size}, D{drafter.max_draft_len})]")

    debug_lines = []

    def append_debug(line):
        debug_lines.append(line)
        del debug_lines[:-80]

    sch = Scheduler(
        eng, drafter, eos_token_ids=tokenizer.eos_token_ids,
        k=drafter.max_draft_len, chunk=512, debug=True,
        dynamic_depth=False, prefix_cache="0",
        output_decode=lambda ids: decode(tokenizer, ids, skip_special_tokens=False),
        log=append_debug,
    )

    prompts = {}
    produced_text = {}
    max_tokens = 8192

    output_buf = Buffer(read_only=True)
    log_buf = Buffer(read_only=True)

    def render():
        output_lines = [
            "[Type a prompt + Enter. Add more while it runs (they join the "
            "batch). :q quits.]",
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
        """Prefill one request and merge it into the live batch (concurrent —
        DFlash batches, so new prompts join a running decode)."""
        rid = next_rid[0]
        next_rid[0] += 1
        prompts[rid] = text
        produced_text[rid] = ""
        group = PrefillGroup(
            req=Req(rid, to_ids(tokenizer, text, False), max_tokens)
        )
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

    render()

    app = Application(layout=layout, key_bindings=kb, full_screen=True,
                      mouse_support=True, refresh_interval=0.1)

    async def driver():
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


def main() -> int:
    # No CLI args -> interactive chat (like try_engine). Any args -> the original
    # one-shot reference smoke test.
    if len(sys.argv) == 1:
        return main_interactive()
    return main_reference(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
