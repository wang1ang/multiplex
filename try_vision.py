"""CLI harness (L5-ish) for Gemma 4 vision in multiplex.

Feeds an image path + a text question directly (no server, no OpenAI API yet):
renders the chat template, expands the ``<|image|>`` placeholder to
BOI+soft-tokens+EOI, encodes the image with the L4 VisionEncoder, splices
text+image embeds, and drives the Scheduler to completion via the Req.embeds
prefill path. MTP (if a head is found) still accelerates decode.

Usage:
    python try_vision.py --image /path/to.png --prompt "What is in this image?"
    python try_vision.py --text-only --prompt "hi"   # regression: no image
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

from multiplex.kernel.engine import Engine
from multiplex.kernel.mtp import find_drafter
from multiplex.kernel.scheduler import Scheduler, Req, PrefillGroup
from multiplex.kernel import vision as V


def _apply_template(tokenizer, text, has_image):
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True)


def run(model_path, image, prompt, max_tokens, k, text_only,
        dynamic_depth=True):
    eng = Engine(model_path)
    tok = eng.tokenizer
    drafter = find_drafter(eng)
    print(f"[loaded; MTP={'yes' if drafter else 'no'}]", file=sys.stderr)

    sch = Scheduler(eng, drafter, eos_token_ids=tok.eos_token_ids,
                    k=k, chunk=512, debug=False,
                    dynamic_depth=dynamic_depth)

    ids = list(_apply_template(tok, prompt, has_image=not text_only))

    embeds = None
    if not text_only:
        enc = V.VisionEncoder(model_path)
        t0 = time.time()
        feats, n = enc.encode(image)
        mx.eval(feats)
        print(f"[encoded image: {n} soft tokens in {time.time()-t0:.2f}s]",
              file=sys.stderr)
        ids = V.expand_image_placeholders(ids, [n])
        embeds = V.build_prefill_embeds(eng.model, ids, [feats])
        mx.eval(embeds)
        print(f"[prompt len {len(ids)} (image tokens={n}); embeds {embeds.shape}]",
              file=sys.stderr)
    else:
        print(f"[text-only prompt len {len(ids)}]", file=sys.stderr)

    req = Req(0, ids, max_tokens, temperature=None, embeds=embeds)
    group = PrefillGroup(req=req)
    while True:
        done = sch.prefill_chunk(group)
        if done:
            break
    sch.merge_ready(group)

    out_ids = list(req.out)
    while sch.has_rows():
        emitted = sch.step()
        for rid, toks in emitted:
            out_ids.extend(toks)

    text = tok.decode(out_ids, skip_special_tokens=True)
    print("\n=== OUTPUT ===")
    print(text)
    return text


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/yang.wang/.mtplx/models/"
                    "Gemma-4-12B-MTPLX-6bit-mtp4bit")
    ap.add_argument("--image", default="/tmp/vtest.png")
    ap.add_argument("--prompt", default="Describe this image. What color and "
                    "shape do you see?")
    ap.add_argument("-n", "--max-tokens", type=int, default=128)
    ap.add_argument(
        "-k",
        "--depth",
        type=int,
        default=3,
        help="maximum dynamic MTP depth (default: 3); fixed with "
             "--no-dynamic-depth; 0 = pure AR",
    )
    ap.add_argument(
        "--dynamic-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="adapt D1..Dmax from live full-depth acceptance (default: enabled)",
    )
    ap.add_argument("--text-only", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.image, args.prompt, args.max_tokens, args.depth,
        args.text_only, args.dynamic_depth)
