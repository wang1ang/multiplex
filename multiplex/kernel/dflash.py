"""L2 - DFlash block-diffusion speculative decoding.

The vendored reference (``DFlashDraftModel`` + ``stream_generate``) is a faithful
copy of the official DFlash MLX code
(z-lab/dflash `dflash/model_mlx.py`, MIT License, Copyright (c) 2026 Z Lab),
kept bit-for-bit; its ``stream_generate`` loop is single-sequence. Below it,
``DFlashDrafter`` adapts the model to multiplex's batched scheduler (B>1). Local
edits to the vendored part are limited to this header and ``load_draft``
accepting a local directory in addition to a Hugging Face repo id.

DFlash drafts a whole block in ONE parallel forward (mask tokens), not the
AR-chained single-token drafting used by ``kernel/mtp.py``'s ``Drafter``. Its
draft model reuses the target's ``embed_tokens`` + ``lm_head`` and conditions on
the target's hidden states at ``target_layer_ids`` (fused by ``fc``). See
``docs/DFLASH.md`` for how this plugs into multiplex's engine/scheduler.

Upstream: https://github.com/z-lab/dflash  (paper: arXiv:2602.06036)
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.generate import generation_stream
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache, can_trim_prompt_cache, make_prompt_cache
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

try:
    import mlx_lm.models.gated_delta as _gd_mod
    _HAS_GDN = True
except ImportError:
    _HAS_GDN = False


_GDN_PATCH_LOCK = RLock()


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: Tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    layer_types: Tuple[str, ...] = field(default_factory=tuple)
    sliding_window: Optional[int] = None
    final_logit_softcapping: Optional[float] = None


def _build_rope(
    head_dim: int,
    rope_theta: float,
    max_position_embeddings: int,
    rope_scaling: Optional[Dict[str, Any]],
):
    return initialize_rope(
        dims=head_dim,
        base=rope_theta,
        traditional=False,
        scaling_config=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(dim, n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        if self.is_sliding:
            keep_ctx = self.sliding_window - 1
            if S > keep_ctx:
                skip = S - keep_ctx
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        mask = None
        if self.is_sliding:
            mask = (
                "causal" if ctx_len + L <= self.sliding_window
                else create_causal_mask(L, offset=ctx_len, window_size=self.sliding_window)
            )
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ("full_attention",) * self.config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(
            config.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
            config.rope_scaling,
        )
        self.embed_tokens = None
        self.lm_head = None
        self.embed_scale = 1.0

    def bind(self, target_model):
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            inner = target_model.model
        elif (hasattr(target_model, "language_model") and
              hasattr(target_model.language_model, "model") and
              hasattr(target_model.language_model.model, "embed_tokens")):
            inner = target_model.language_model.model
        else:
            raise AttributeError(f"Cannot find embed_tokens in {type(target_model).__name__}")
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0))
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = getattr(target_model, "lm_head", None) or getattr(lm, "lm_head", None) or self.embed_tokens.as_linear
        return self

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError("Draft config must define sliding_window for sliding_attention layers.")
                caches.append(RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0))
            else:
                caches.append(KVCache())
        return caches

    def __call__(
        self,
        inputs,
        target_hidden,
        cache,
        logits_start: int = 0,
    ):
        h = self.embed_tokens(inputs) * self.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        if logits_start:
            h = h[:, logits_start:]
        logits = self.lm_head(self.norm(h))
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits


def load(model_id: str):
    from mlx_lm import load as mlx_lm_load
    return mlx_lm_load(model_id)


def load_draft(draft_id: str) -> DFlashDraftModel:
    # Accept a local directory (multiplex resolves models on disk) or fall back
    # to a Hugging Face repo id (reference behaviour).
    if os.path.isdir(os.path.expanduser(draft_id)):
        path = Path(os.path.expanduser(draft_id))
    else:
        path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    cfg = json.loads((path / "config.json").read_text())
    layer_types = tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])
    if len(layer_types) != cfg["num_hidden_layers"]:
        raise ValueError("Draft config layer_types length must match num_hidden_layers.")
    unknown_layer_types = set(layer_types) - {"full_attention", "sliding_attention"}
    if unknown_layer_types:
        raise ValueError(f"Unsupported draft layer_types: {sorted(unknown_layer_types)}.")
    if "sliding_attention" in layer_types and cfg.get("sliding_window") is None:
        raise ValueError("Draft config must define sliding_window for sliding_attention layers.")
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=cfg["block_size"],
        target_layer_ids=tuple(cfg["dflash_config"]["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=cfg["dflash_config"]["mask_token_id"],
        rope_scaling=cfg.get("rope_scaling"),
        layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=cfg.get("final_logit_softcapping"),
    )
    weights = {k: v for f in path.glob("*.safetensors") for k, v in mx.load(str(f)).items()}
    model = DFlashDraftModel(config)
    model.load_weights(list(weights.items()))
    return model


def _trim_recent_cache(cache: List[Any], num_tokens: int) -> None:
    if num_tokens <= 0:
        return
    for c in cache:
        n = min(getattr(c, "offset", num_tokens), num_tokens)
        if n <= 0:
            continue
        if isinstance(c, RotatingKVCache) and c.keys is not None:
            c.keys = c._temporal_order(c.keys)
            c.values = c._temporal_order(c.values)
            c.keys = c.keys[..., :-n, :]
            c.values = c.values[..., :-n, :]
            c.offset -= n
            c._idx = c.keys.shape[2]
        elif hasattr(c, "trim"):
            c.trim(n)


class _LayerHook:
    def __init__(self, layer, idx, storage):
        self._layer, self._idx, self._storage = layer, idx, storage

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        self._storage[self._idx] = out[0] if isinstance(out, tuple) else out
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


def _patch_model(model, layer_ids):
    if hasattr(model, "_hidden_states"):
        return
    model._hidden_states = [None] * len(layer_ids)
    layers = _get_layers(model)
    for i, lid in enumerate(layer_ids):
        layers[lid] = _LayerHook(layers[lid], i, model._hidden_states)


class _GDNStateCapture:
    def __init__(self):
        self.conv_data = []
        self._gdn_inputs = []
        self._gdn_cls = None
        self._orig_call = None
        self._patched_call = None
        self._closed = False
        _GDN_PATCH_LOCK.acquire()
        try:
            self._patch()
        except Exception:
            _GDN_PATCH_LOCK.release()
            raise

    def _patch(self):
        from mlx_lm.models.qwen3_5 import GatedDeltaNet
        self._gdn_cls = GatedDeltaNet
        self._orig_call = GatedDeltaNet.__call__
        capture = self

        def _capturing_gdn_call(self_layer, inputs, mask=None, cache=None):
            B, S, _ = inputs.shape
            if self_layer.sharding_group is not None:
                from mlx_lm.models.qwen3_5 import sum_gradients
                inputs = sum_gradients(self_layer.sharding_group)(inputs)
            qkv = self_layer.in_proj_qkv(inputs)
            z = self_layer.in_proj_z(inputs).reshape(B, S, self_layer.num_v_heads, self_layer.head_v_dim)
            b, a = self_layer.in_proj_b(inputs), self_layer.in_proj_a(inputs)
            conv_state = cache[0] if (cache is not None and cache[0] is not None) else mx.zeros((B, self_layer.conv_kernel_size - 1, self_layer.conv_dim), dtype=inputs.dtype)
            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            capture.conv_data.append((conv_input, self_layer.conv_kernel_size))
            if cache is not None:
                cache[0] = conv_input[:, -(self_layer.conv_kernel_size - 1):]
            conv_out = nn.silu(self_layer.conv1d(conv_input))
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self_layer.key_dim, 2 * self_layer.key_dim], -1),
                    [self_layer.num_k_heads, self_layer.num_k_heads, self_layer.num_v_heads],
                    [self_layer.head_k_dim, self_layer.head_k_dim, self_layer.head_v_dim],
                )
            ]
            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            capture._gdn_inputs.append((q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask))
            out, new_state = _gd_mod.gated_delta_update(
                q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask, use_kernel=True
            )
            if cache is not None:
                cache[1] = new_state
            out = self_layer.norm(out, z)
            out = self_layer.out_proj(out.reshape(B, S, -1))
            if self_layer.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self_layer.sharding_group)
            return out

        self._patched_call = _capturing_gdn_call
        GatedDeltaNet.__call__ = _capturing_gdn_call

    def clear(self):
        self.conv_data.clear()
        self._gdn_inputs.clear()

    def close(self):
        if self._closed:
            return
        try:
            if self._gdn_cls is not None and self._gdn_cls.__call__ is self._patched_call:
                self._gdn_cls.__call__ = self._orig_call
        finally:
            self._closed = True
            self._gdn_cls = None
            self._orig_call = None
            self._patched_call = None
            _GDN_PATCH_LOCK.release()

    def rollback(self, cache, accepted, trim):
        n_non_trimmable = sum(1 for c in cache if not c.is_trimmable())
        assert n_non_trimmable == len(self._gdn_inputs), (
            f"non-trimmable cache count ({n_non_trimmable}) != "
            f"captured GDN inputs ({len(self._gdn_inputs)}); "
            "DFlash MLX rollback assumes every non-trimmable cache is a GatedDeltaNet layer"
        )
        j = 0
        for c in cache:
            if c.is_trimmable():
                c.trim(trim)
            else:
                q, k, v, a, b, A_log, dt_bias, init_state, mask = self._gdn_inputs[j]
                n = accepted + 1
                _, state = _gd_mod.gated_delta_update(
                    q[:, :n], k[:, :n], v[:, :n], a[:, :n], b[:, :n],
                    A_log, dt_bias, init_state,
                    None if mask is None else mask[:, :n],
                    use_kernel=True,
                )
                c.cache[1] = state
                conv_input, K = self.conv_data[j]
                c.cache[0] = conv_input[:, accepted + 1 : accepted + K]
                j += 1


@dataclass
class GenerationResponse:
    text: str
    tokens: List[int]
    accepted: int
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None


def _make_response(
    text,
    tokens,
    accepted,
    prompt_size,
    prompt_tps,
    n,
    tic,
    finish_reason=None,
):
    return GenerationResponse(
        text, tokens, accepted, prompt_size, prompt_tps,
        n, n / (time.perf_counter() - tic), mx.get_peak_memory() / 1e9, finish_reason,
    )


def stream_generate(
    model, draft, tokenizer, prompt,
    block_size=None, max_tokens=256, temperature=0.0, sampler=None,
):
    _patch_model(model, draft.config.target_layer_ids)
    block_size = block_size if block_size is not None else int(draft.config.block_size)
    sampler = sampler or make_sampler(temp=temperature)

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer
    mask_id = int(draft.config.mask_token_id)
    tokens = prompt.tolist()

    target_cache = make_prompt_cache(model)
    draft_cache = make_prompt_cache(draft)
    draft.bind(model)
    _target_can_trim = can_trim_prompt_cache(target_cache)
    if not _target_can_trim and not _HAS_GDN:
        raise RuntimeError(
            "This MLX model requires gated-delta rollback support, but "
            "mlx_lm.models.gated_delta is unavailable."
        )
    _capture = _GDNStateCapture() if not _target_can_trim else None

    try:
        tic = time.perf_counter()
        with mx.stream(generation_stream):
            logits = model(prompt[None], target_cache)
            hidden = mx.concatenate(model._hidden_states, axis=-1)
        mx.eval(logits, hidden)
        prompt_tps = prompt.size / (time.perf_counter() - tic)

        tic = time.perf_counter()
        token = sampler(logits[:, -1:])[0, 0].item()
        tokens.append(token)
        n = 1

        if token in tokenizer.eos_token_ids:
            detokenizer.add_token(token)
            detokenizer.finalize()
            yield _make_response(detokenizer.last_segment, [token], 1, prompt.size, prompt_tps, n, tic, "stop")
            return

        detokenizer.add_token(token)
        yield _make_response(
            detokenizer.last_segment,
            [token],
            1,
            prompt.size,
            prompt_tps,
            n,
            tic,
        )

        while n < max_tokens:
            bs = min(block_size, max_tokens - n + 1)
            if bs <= 1:
                break

            with mx.stream(generation_stream):
                block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
                draft_logits = draft(
                    block,
                    hidden,
                    draft_cache,
                    logits_start=1,
                )
                if (trim_n := draft_cache[0].offset - (prompt.size + n - 1)) > 0:
                    _trim_recent_cache(draft_cache, trim_n)
                draft_tokens = sampler(draft_logits)
            mx.async_eval(draft_tokens)

            if _capture is not None:
                _capture.clear()
            with mx.stream(generation_stream):
                verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens], axis=1)
                logits = model(verify_input, target_cache)
                hidden = mx.concatenate(model._hidden_states, axis=-1)
                target_tokens = sampler(logits)
            mx.async_eval(target_tokens, hidden)

            d_list, t_list = draft_tokens[0].tolist(), target_tokens[0].tolist()
            accepted = next((i for i in range(len(d_list)) if d_list[i] != t_list[i]), len(d_list))
            new_tokens = d_list[:accepted] + [t_list[accepted]]
            new_tokens = new_tokens[:max_tokens - n]

            eos_idx = next((i for i, t in enumerate(new_tokens) if t in tokenizer.eos_token_ids), None)
            if eos_idx is not None:
                new_tokens = new_tokens[:eos_idx + 1]
                for t in new_tokens:
                    detokenizer.add_token(t)
                detokenizer.finalize()
                tokens.extend(new_tokens)
                n += len(new_tokens)
                yield _make_response(
                    detokenizer.last_segment,
                    new_tokens,
                    accepted + 1,
                    prompt.size,
                    prompt_tps,
                    n,
                    tic,
                    "stop",
                )
                return

            for t in new_tokens:
                detokenizer.add_token(t)
            tokens.extend(new_tokens)
            n += len(new_tokens)

            if n % 256 == 0:
                mx.clear_cache()

            yield _make_response(
                detokenizer.last_segment,
                new_tokens,
                accepted + 1,
                prompt.size,
                prompt_tps,
                n,
                tic,
            )

            trim = bs - accepted - 1
            if trim > 0:
                if _target_can_trim:
                    _trim_recent_cache(target_cache, trim)
                elif _capture is not None:
                    _capture.rollback(target_cache, accepted, trim)
            hidden = hidden[:, :accepted + 1, :]

        detokenizer.finalize()
        yield _make_response(
            detokenizer.last_segment,
            [],
            0,
            prompt.size,
            prompt_tps,
            n,
            tic,
            "length",
        )
    finally:
        if _capture is not None:
            _capture.close()


# ---------------------------------------------------------------------------
# multiplex integration: DFlash as a drafter behind the generic contract
# (see docs/ARCHITECTURE.md). This is NOT part of the vendored reference.
# ---------------------------------------------------------------------------


class DFlashDrafter:
    """DFlash block-diffusion drafter behind multiplex's generic drafter contract.

    Unlike MTP (AR-chained, drafts one token at a time from the trunk's FINAL
    hidden, single-layer draft cache), DFlash drafts a whole block in ONE
    parallel forward, conditioned on the trunk's TAPPED hidden states
    (``config.target_layer_ids``), with its own multi-layer KV cache. It reads
    the taps from the engine (``engine.last_hidden_taps``, populated when
    ``engine.enable_hidden_taps`` is on), not the final hidden.

    ``draft(ctx, primary, k, cache)`` feeds ``ctx`` (the taps of the positions
    committed since the last draft — the whole prompt for the first draft) as the
    draft model's cross-attention context; recording it there IS DFlash's
    committed-history append, so the scheduler needs no separate append/trim on
    the commit path.

    Batched decode (B>1) is supported: each row keeps its own 5-layer draft
    cache and its own context (a per-row list), and ``draft()`` loops the draft
    model over rows. This reuses the validated single-sequence attention and
    sidesteps a batched/left-padded rewrite; the per-row draft forwards are not
    yet parallelized (see docs/DFLASH.md). Prefix-cache reuse of the draft ctx
    is not supported (``supports_prefix_reuse=False``).
    """

    wants_taps = True
    supports_dynamic_depth = False   # trained for a fixed block; k is pinned
    # ...but the fixed-width block can be VERIFIED at an adaptive (smaller)
    # prefix: the draft cache only ever stores committed context (the block's
    # own K/V is never written — see DFlashAttention), so shrinking the trunk
    # verify width needs no draft-cache rollback. The scheduler drafts the full
    # block and verifies self.k <= max_draft_len of it.
    supports_adaptive_verify = True
    supports_prefix_reuse = False
    supports_batching = True

    def __init__(self, engine, model: DFlashDraftModel):
        self.engine = engine
        self.model = model
        model.bind(engine.model)
        self.tap_layer_ids = tuple(int(i) for i in model.config.target_layer_ids)
        self.block_size = int(model.config.block_size)
        self.mask_id = int(model.config.mask_token_id)

    @property
    def max_draft_len(self) -> int:
        # A block is ``[committed_token, mask*(block_size-1)]``; the masked tail
        # is what gets drafted, so the draft depth is block_size - 1.
        return self.block_size - 1

    def make_cache(self) -> list:
        # The multi-row draft cache is a list with one entry per batch row; each
        # entry is the draft model's own 5-layer cache. Rows are kept separate
        # (not fused into BatchKVCache) so each row's draft forward uses the
        # validated single-sequence attention path, looped in draft().
        return [self.model.make_cache()]

    def _taps(self, engine):
        """The trunk's tapped hiddens from the latest forward (the engine stashed
        them); DFlash conditions the draft on these."""
        taps = engine.last_hidden_taps
        if taps is None:
            raise RuntimeError(
                "DFlash drafter needs engine hidden taps; call "
                "engine.enable_hidden_taps(tap_layer_ids) before serving.")
        return taps

    def update_prefill_context(self, engine, hidden, previous):
        # Per-row context is a one-element list ``[taps]`` while a single request
        # prefills; chunks accumulate on the sequence axis.
        taps = self._taps(engine)
        if previous is None:
            return [taps]
        return [mx.concatenate([previous[0], taps], axis=1)]

    def update_context_after_commit(self, engine, hidden, accepted):
        # Return the committed positions' taps PER ROW (ragged across rows is
        # fine: draft() consumes each row separately).
        taps = self._taps(engine)
        accepts = ([int(accepted)] * int(taps.shape[0])
                   if isinstance(accepted, int) else accepted)
        return [
            taps[i:i + 1, :int(a) + 1, :]
            for i, a in enumerate(accepts)
        ]

    def merge_context(self, ctxs):
        # Each element is a per-row context list; flatten into one list of rows.
        return [row for sub in ctxs for row in sub]

    def filter_context(self, ctx, keep):
        return [ctx[i] for i in keep]

    def draft(self, ctx, primary, k, cache) -> mx.array:
        if k == 0:
            return mx.zeros((primary.shape[0], 0), dtype=primary.dtype)
        # Draft each row on its own 5-layer cache (single-sequence path). The
        # draft model is small; looping B times is cheap next to the batched
        # trunk verify, and it sidesteps a batched/left-padded rewrite of the
        # ctx/prop-split attention.
        outs = []
        for i in range(len(cache)):
            prim = primary[i:i + 1]                                   # [1]
            masks = mx.full((1, k), self.mask_id, dtype=primary.dtype)
            block = mx.concatenate([prim[:, None], masks], axis=1)    # [1, 1+k]
            logits = self.model(block, ctx[i], cache[i], logits_start=1)  # [1,k,V]
            outs.append(mx.argmax(logits, axis=-1))                   # [1, k]
        return mx.concatenate(outs, axis=0)                          # [B, k]

    def commit(self, cache, m, draft_ids, hidden) -> None:
        # No-op: DFlash records committed context inside the NEXT draft() (it
        # feeds the committed taps as cross-attention context) and never writes
        # the drafted block to its cache, so there is nothing to trim or append.
        pass

    # --- batch lifecycle: per-row draft caches (list, one entry per row) -----
    def extract_cache_row(self, cache, i):
        return [cache[i]]

    def merge_caches(self, caches):
        return [row for dc in caches for row in dc]

    def filter_cache(self, cache, keep):
        cache[:] = [cache[i] for i in keep]

    # --- prefix cache: reuse of the draft ctx is not supported in v1 ---------
    def clone_cache_block(self, cache, start, pos, *, row=None):
        return [None]

    def restore_cache_blocks(self, blocks):
        return self.make_cache()


def find_dflash(model_path: str) -> str | None:
    """Return the target's DFlash draft sidecar directory, or None.

    Mirrors ``mtp.find_mtp``: a DFlash-enabled target ships its matched draft as
    a ``dflash/`` subdir (config.json + weights) beside the model, so the draft
    travels with the model and needs no hardcoded, machine-specific path."""
    sidecar = os.path.join(os.path.expanduser(model_path), "dflash")
    if os.path.isdir(sidecar) and os.path.exists(os.path.join(sidecar, "config.json")):
        return sidecar
    return None


def build_dflash_drafter(engine, draft_dir: str) -> DFlashDrafter:
    """Load a DFlash draft model from ``draft_dir`` and wrap it as a drafter
    bound to ``engine``'s trunk. The caller must also enable the engine's hidden
    taps for ``drafter.tap_layer_ids``."""
    return DFlashDrafter(engine, load_draft(draft_dir))
