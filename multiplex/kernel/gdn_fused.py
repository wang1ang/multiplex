"""Fused GDN (GatedDeltaNet) kernels for the decode / verify widths.

Adapted from MTPLX's mtplx/kernels/gdn_conv_norm.py (rows variant) and
gdn_step_fused.py, re-derived for mlx-lm's qwen3_5 GatedDeltaNet numerics
(the MTPLX originals target a sigmoid-gated, l2norm family):

  * q/k norm is mlx-lm's ``(1/D) * rms_norm(q)`` / ``D^-0.5 * rms_norm(k)``
    (eps on the MEAN of squares, scale applied as a bf16 scalar), not l2norm;
  * the output norm gate is silu (Qwen3NextRMSNormGated), not sigmoid;
  * every bf16 rounding point of the stock chain (conv1d out, sigmoid, silu
    product, rms_norm out, scale, softplus, beta, recurrence out) is kept, so
    outputs match the stock chain up to fp32 summation order.

Two kernels, both batched over B rows (one conv/delta state per row):

  conv_norm_rows   S >= 2 rows (verify / replay / copy blocks):
                   conv1d(k=4) + silu + q/k norm in ONE dispatch, replacing
                   concat/conv/silu/split/2x rms_norm/2x scale; the recurrence
                   stays in mlx-lm's gated_delta_update kernel.
  step             S == 1 (AR decode, m=0 replay): conv + silu + q/k norm +
                   g/beta + delta recurrence + gated-norm epilogue in ONE
                   dispatch between the in_proj and out_proj GEMVs.

Threadgroups are 128 (rows) / 256 (step) threads, so the 1024-thread
pipeline-cap problem MTPLX probes for on M2/M3-class GPUs does not arise.
Geometry: head dim 128 and conv kernel 4 are fixed; head counts are template
parameters. ``unsupported_reason`` decides per model; the per-call gate only
routes shapes (masked / ragged / prefill-width calls keep the stock chain).
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

D = 128
CONV_K = 4
MAX_ROWS = 64          # verify/copy blocks are <= 33; prefill pieces stay stock

_HEADER = """
#include <metal_stdlib>
using namespace metal;

// mlx's Sigmoid functor (unary_ops.h) evaluated in T: every op rounds to T.
template <typename T> inline float sigmoid_T(float xf) {
  const T x = (T)xf;
  const T y = 1 / (1 + metal::exp(metal::abs(x)));
  return (float)((x < 0) ? y : (T)(1 - y));
}
// mlx's softplus = logaddexp(x, 0) (binary_ops.h + bf16 log1p, utils.h),
// evaluated in T like the compiled stock op.
template <typename T> inline float softplus_T(T x) {
  const T mxv = metal::max(x, T(0));
  const T mnv = metal::min(x, T(0));
  const T e = metal::exp(mnv - mxv);
  const float xp1 = 1.0f + (float)e;
  const T l = (xp1 == 1.0f) ? e : (T)((float)e * (metal::log(xp1) / (xp1 - 1.0f)));
  return (float)(T)(mxv + l);
}
// Sum of squares of a 128-wide head held one value per thread in vals[],
// in mlx rms_single_row's order (32 threads x 4 adjacent reads, simd_sum).
// Call from all threads of the head's 4 simdgroups; sg0 = first of them.
inline float head_sumsq(threadgroup float* vals, threadgroup float* out,
                        uint sg, uint sg0, uint lane) {
  if (sg == sg0) {
    float acc = 0;
    for (int i = 0; i < 4; i++) {
      const float x = vals[lane * 4 + i];
      acc += x * x;
    }
    acc = simd_sum(acc);
    if (lane == 0) out[0] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out[0];
}
"""

# One threadgroup per head (128 channels); grid.y = batch row.
_ROWS_SRC = """
    constexpr int QK = HK * 128;
    constexpr int C = (2 * HK + HV) * 128;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint head = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint c = head * 128 + tid;
    const int S = n_rows;
    threadgroup float vals[128];
    threadgroup float tot[1];

    const device T* xb = xnew + b * S * C;
    const device T* sb = state + b * 3 * C;
    const float w0 = (float)cw[c * 4 + 0];
    const float w1 = (float)cw[c * 4 + 1];
    const float w2 = (float)cw[c * 4 + 2];
    const float w3 = (float)cw[c * 4 + 3];
    #define STREAM(t) ((t) < 3 ? (float)sb[(t) * C + c] : (float)xb[((t) - 3) * C + c])
    const bool is_v = head >= 2 * HK;
    const float qs = (float)(T)(1.0f / 128.0f);
    const float ks = (float)(T)(0.08838834764831845f);

    for (int s = 0; s < S; ++s) {
      const float x = (float)(T)(w0 * STREAM(s) + w1 * STREAM(s + 1)
                                 + w2 * STREAM(s + 2) + w3 * STREAM(s + 3));
      const float sv = (float)(T)(x * sigmoid_T<T>(x));
      if (is_v) {
        v_out[(b * S + s) * (HV * 128) + (c - 2 * QK)] = (T)sv;
        continue;
      }
      vals[tid] = sv;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float inv = metal::precise::rsqrt(
          head_sumsq(vals, tot, sg, 0, lane) / 128.0f + 1e-6f);
      const float n = (float)(T)(sv * inv);
      if (head < HK) {
        q_out[(b * S + s) * QK + c] = (T)(n * qs);
      } else {
        k_out[(b * S + s) * QK + (c - QK)] = (T)(n * ks);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device T* so = state_out + b * 3 * C;
    so[0 * C + c] = (T)STREAM(S + 0);
    so[1 * C + c] = (T)STREAM(S + 1);
    so[2 * C + c] = (T)STREAM(S + 2);
    #undef STREAM
"""

# One threadgroup (256 threads, 8 simdgroups) per (value head, batch row).
_STEP_SRC = """
    constexpr int QK = HK * 128;
    constexpr int C = (2 * HK + HV) * 128;
    constexpr uint R = HV / HK;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint b = threadgroup_position_in_grid.y;
    const uint hv = threadgroup_position_in_grid.z;
    const uint hk = hv / R;

    threadgroup float qn[128];
    threadgroup float kn[128];
    threadgroup float vv[128];
    threadgroup float outv[128];
    threadgroup float tot[3];

    const device T* xb = xnew + b * C;
    const device T* sb = state + b * 3 * C;
    device T* so = state_out + b * 3 * C;

    // ---- conv + silu over this head's q | k | v channels ----
    for (uint i = tid; i < 384; i += 256) {
      uint g;
      if (i < 128) g = hk * 128 + i;
      else if (i < 256) g = QK + hk * 128 + (i - 128);
      else g = 2 * QK + hv * 128 + (i - 256);
      const float x = (float)(T)(
          (float)cw[g * 4 + 0] * (float)sb[0 * C + g] +
          (float)cw[g * 4 + 1] * (float)sb[1 * C + g] +
          (float)cw[g * 4 + 2] * (float)sb[2 * C + g] +
          (float)cw[g * 4 + 3] * (float)xb[g]);
      const float sv = (float)(T)(x * sigmoid_T<T>(x));
      // each conv channel has exactly one state writer
      if (i >= 256 || (hv % R) == 0) {
        so[0 * C + g] = sb[1 * C + g];
        so[1 * C + g] = sb[2 * C + g];
        so[2 * C + g] = xb[g];
      }
      if (i < 128) qn[i] = sv;
      else if (i < 256) kn[i - 128] = sv;
      else vv[i - 256] = sv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- q/k: (1/D)*rms_norm(q), D^-0.5*rms_norm(k) ----
    {
      const float x = (tid < 128) ? qn[tid] : kn[tid - 128];
      const float qinv = metal::precise::rsqrt(
          head_sumsq(qn, tot + 0, sg, 0, lane) / 128.0f + 1e-6f);
      const float kinv = metal::precise::rsqrt(
          head_sumsq(kn, tot + 1, sg, 4, lane) / 128.0f + 1e-6f);
      if (tid < 128) {
        qn[tid] = (float)(T)((float)(T)(x * qinv) * (float)(T)(1.0f / 128.0f));
      } else {
        kn[tid - 128] = (float)(T)((float)(T)(x * kinv)
                                   * (float)(T)(0.08838834764831845f));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b) ----
    const float sp = softplus_T<T>((T)(a_row[b * HV + hv] + dt_bias[hv]));
    // compiled mlx graphs (compute_g, _precise_swiglu) use precise exp
    const float g_dec = metal::precise::exp(-metal::precise::exp((float)A_log[hv]) * sp);
    const float beta = sigmoid_T<T>((float)b_row[b * HV + hv]);

    // ---- delta recurrence (mirrors mlx-lm gated_delta_step) ----
    const uint sbase = (b * HV + hv) * 128 * 128;
    for (uint dv = sg; dv < 128; dv += 8) {
      const uint srow = sbase + dv * 128 + 4 * lane;
      float st[4];
      float kv_mem = 0.0f;
      for (int i = 0; i < 4; ++i) {
        st[i] = (float)dstate[srow + i] * g_dec;
        kv_mem += st[i] * kn[4 * lane + i];
      }
      kv_mem = simd_sum(kv_mem);
      const float delta = (vv[dv] - kv_mem) * beta;
      float out = 0.0f;
      for (int i = 0; i < 4; ++i) {
        st[i] = st[i] + kn[4 * lane + i] * delta;
        out += st[i] * qn[4 * lane + i];
        dstate_out[srow + i] = (StT)st[i];
      }
      out = simd_sum(out);
      if (lane == 0) outv[dv] = (float)(T)out;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Qwen3NextRMSNormGated: silu(z) * rms_norm(out, w) ----
    {
      const float o = (tid < 128) ? outv[tid] : 0.0f;
      const float rinv = metal::precise::rsqrt(
          head_sumsq(outv, tot + 2, sg, 0, lane) / 128.0f + eps);
      if (tid < 128) {
        const float xn = (float)(T)((float)norm_w[tid] * (float)(T)(o * rinv));
        const float zf = (float)z_row[(b * HV + hv) * 128 + tid];
        const float y0 = 1.0f / (1.0f + metal::precise::exp(metal::abs(zf)));
        const float gate = zf * ((zf < 0.0f) ? y0 : 1.0f - y0);
        y[(b * HV + hv) * 128 + tid] = (T)(gate * xn);
      }
    }
"""


@lru_cache(maxsize=None)
def _rows_kernel():
    return mx.fast.metal_kernel(
        name="multiplex_gdn_conv_norm_rows",
        input_names=["xnew", "state", "cw", "n_rows"],
        output_names=["q_out", "k_out", "v_out", "state_out"],
        header=_HEADER, source=_ROWS_SRC)


@lru_cache(maxsize=None)
def _step_kernel():
    return mx.fast.metal_kernel(
        name="multiplex_gdn_step",
        input_names=["xnew", "z_row", "a_row", "b_row", "state", "cw",
                     "A_log", "dt_bias", "dstate", "norm_w", "eps"],
        output_names=["y", "state_out", "dstate_out"],
        header=_HEADER, source=_STEP_SRC)


def unsupported_reason(mixer) -> str | None:
    """Why ``mixer`` (an mlx-lm qwen3_5/qwen3_next GatedDeltaNet) can't use
    these kernels, or None if it can."""
    from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated
    checks = [
        (mixer.head_k_dim == D and mixer.head_v_dim == D,
         f"head dims {mixer.head_k_dim}/{mixer.head_v_dim} != {D}"),
        (mixer.conv_kernel_size == CONV_K, f"conv kernel {mixer.conv_kernel_size} != {CONV_K}"),
        (mixer.num_v_heads % mixer.num_k_heads == 0, "num_v_heads not a multiple of num_k_heads"),
        (getattr(mixer.conv1d, "bias", None) is None, "conv1d has a bias"),
        (type(mixer.norm) is Qwen3NextRMSNormGated, f"output norm is {type(mixer.norm).__name__}"),
        (getattr(mixer, "sharding_group", None) is None, "mixer is sharded"),
        (mixer.conv1d.weight.dtype in (mx.bfloat16, mx.float16), f"conv dtype {mixer.conv1d.weight.dtype}"),
    ]
    for ok, why in checks:
        if not ok:
            return why
    return None


def applies(inputs: mx.array, mask, cache) -> bool:
    """Shape routing: a live (post-prefill) cache, no mask / ragged lengths,
    and a decode/verify width. Everything else runs the stock chain."""
    S = inputs.shape[1]
    return (cache is not None and mask is None and S <= MAX_ROWS
            and getattr(cache, "lengths", None) is None
            and cache[0] is not None and cache[1] is not None
            and cache[1].dtype == mx.float32)


def forward(m, inputs: mx.array, cache, record: dict | None = None) -> mx.array:
    """GatedDeltaNet.__call__ for an ``applies`` call, using the fused kernels.

    ``record`` (S >= 2 only) receives the projected recurrence inputs so a
    speculative rollback can ``replay`` a committed prefix without re-running
    the in_proj GEMVs or the conv/norm kernel."""
    from mlx_lm.models.gated_delta import gated_delta_update

    B, S, _ = inputs.shape
    Hk, Hv, C = m.num_k_heads, m.num_v_heads, m.conv_dim
    dt = inputs.dtype
    qkv = m.in_proj_qkv(inputs)
    z = m.in_proj_z(inputs)
    b = m.in_proj_b(inputs)
    a = m.in_proj_a(inputs)
    cw = m.conv1d.weight.reshape(-1)
    tmpl = [("T", dt), ("HK", Hk), ("HV", Hv)]

    if S == 1:
        y, conv, delta = _step_kernel()(
            inputs=[qkv.reshape(-1), z.reshape(-1), a.reshape(-1), b.reshape(-1),
                    cache[0].reshape(-1), cw, m.A_log, m.dt_bias,
                    cache[1].reshape(-1), m.norm.weight, float(m.norm.eps)],
            template=tmpl + [("StT", cache[1].dtype)],
            grid=(256, B, Hv), threadgroup=(256, 1, 1),
            output_shapes=[(B, 1, Hv * D), (B, CONV_K - 1, C), cache[1].shape],
            output_dtypes=[dt, dt, cache[1].dtype])
        cache[0], cache[1] = conv, delta
        cache.advance(1)
        return m.out_proj(y)

    q, k, v, conv = _rows_kernel()(
        inputs=[qkv.reshape(-1), cache[0].reshape(-1), cw, S],
        template=tmpl,
        grid=(C, B, 1), threadgroup=(D, 1, 1),
        output_shapes=[(B, S, Hk, D), (B, S, Hk, D), (B, S, Hv, D), (B, CONV_K - 1, C)],
        output_dtypes=[dt] * 4)
    if record is not None:
        record.update(qkv=qkv, q=q, k=k, v=v, a=a, b=b)
    cache[0] = conv
    out, state = gated_delta_update(q, k, v, a, b, m.A_log, m.dt_bias, cache[1], None)
    cache[1] = state
    cache.advance(S)
    out = m.norm(out, z.reshape(B, S, Hv, D))
    return m.out_proj(out.reshape(B, S, -1))


def replay(m, rec: dict, cache, n: int) -> None:
    """Advance a restored GDN cache over the first ``n`` tokens of a recorded
    ``forward``. The result is exactly the state the verify pass itself held
    after its n-th token: the conv state is the last CONV_K-1 raw qkv rows of
    the stream, and the recurrence is prefix-causal (same kernel, same
    per-step inputs). Re-running the mixer instead recomputes in_proj at
    width n, and the quantized GEMV is width-dependent at n == 1 (M=1 vs M>1
    kernels round differently), so that path drifts from the verify pass."""
    from mlx_lm.models.gated_delta import gated_delta_update

    stream = mx.concatenate([cache[0], rec["qkv"][:, :n]], axis=1)
    cache[0] = stream[:, n:n + CONV_K - 1]
    _, state = gated_delta_update(
        rec["q"][:, :n], rec["k"][:, :n], rec["v"][:, :n], rec["a"][:, :n],
        rec["b"][:, :n], m.A_log, m.dt_bias, cache[1], None)
    cache[1] = state
    cache.advance(n)
