# Running on NVIDIA GPUs

multiplex runs on Linux through MLX's CUDA backend (see the README for
`CUDA_HOME`). It works, but it is not fast there yet. On NVIDIA hardware, a
CUDA-native engine is currently the better choice. This page records what we
measured and what the alternatives reach on the same card, as a reference.

## What multiplex reaches

Measured on an NVIDIA L4 (24 GB, about 300 GB/s) with
`mlx-community/Qwen3.8-27B-4bit` and its MTP sidecar:

| Configuration | Decode |
|---|---|
| Speculative decoding off (D0) | 12.8–13.4 tok/s |
| MTP, depth 3 | 12.7 tok/s |

Trunk forward pass by verification width:

| Width | Time |
|---|---|
| 1 | 76 ms |
| 2 | 122 ms |
| 4 | 215 ms |

Peak GPU memory: 16.9 GB.

MTP gains nothing on this card because of three costs:

- **Verification width.** Stock mlx-cuda's 4-bit matmul at small row counts
  costs roughly linearly in rows. Verifying 4 tokens costs almost 3× one
  token, so drafts don't pay for themselves.
- **Fused GDN kernels.** They are Metal shaders, so on CUDA the Gated DeltaNet
  layers take the unfused path.
- **Bandwidth.** The weight read alone limits plain decoding to about
  18 tok/s.

## What other engines reach on the same card

These are a third party's figures for Qwen3.8-27B on a single L4, from
[hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4).
We have not reproduced them. The model is `unsloth/Qwen3.8-27B-GGUF`
`UD-Q4_K_XL` (17.6 GB). The decode figure is the mean of seven workloads.

| Engine and configuration | Decode |
|---|---|
| llama.cpp, no speculation | 15.0 tok/s |
| Lucebox, DDTree tree drafting (best budget) | 25.2 tok/s |
| llama.cpp, native MTP head, verification routed to MMQ | 31.4 tok/s |
| llama.cpp, DFlash2 block drafter, all types on MMQ (shipped) | 37.4 tok/s |

The DFlash2 configuration splits further by workload:

| Workload | Decode |
|---|---|
| Math | 48.2 tok/s |
| Code | 28.4 tok/s |

What makes the difference:

- **Verification on tensor cores.** llama.cpp normally sends batches of up to
  8 tokens to `mul_mat_vec_q`, which runs on CUDA cores. Speculative
  verification batches fall in that range, so they pay per extra token. A
  one-line patch (`GGML_MMVQ_MAX=2`) sends them to `mul_mat_q` instead, which
  uses the int8 tensor cores. There, widths 2 through 16 cost about 2 ms per
  token. This is the same problem multiplex has on mlx-cuda.
- **A block drafter.** DFlash2 drafts a whole block per pass. Each
  verification accepts 3.63 tokens on average.

vLLM and SGLang reach their best figures for this model with NVFP4. The L4 is
an Ada card and has no FP4 support, and the FP8 checkpoint (27 GB) doesn't fit
in 24 GB.

## Closing the gap in multiplex

The kernels already exist in open source. The work is adapting them:

- **4-bit matmul at small widths.** Marlin (vLLM) handles batch 1–16, and
  llama.cpp has MMQ. MLX can load a compiled kernel through
  `mx.fast.cuda_kernel` / `mx.fast.precompiled_cuda_kernel`. MLX's affine
  format (`w = scale·q + bias`, group 64) maps onto Marlin's float-zero-point
  mode with `zp = -bias/scale`. That needs an offline repack, not a requant.
- **Gated DeltaNet.** flash-linear-attention has these kernels, but they are
  written in Triton, which MLX cannot call directly.

Try a newer MLX first. The CUDA backend changes quickly, and small-width
quantized matmul may already be faster than in the version measured above.
