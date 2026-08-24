# 本地 MLX kernel 维护说明

## 目的

`third_party/mlx` 是 multiplex 使用的本地 Python MLX fork。它用于逐个移植
`qwen-3.8-mtp-challenge` 的 Metal kernel 优化，同时保留 MLX 原有实现作为
fallback。

不要把 challenge 的整个 `quantized.h`、`quantized.metal` 或其他文件直接覆盖
到 Python MLX。两个仓库的版本、dispatch、kernel 实例和 batch shape 不完全一致，
整文件覆盖会造成 kernel 缺失、函数重定义或 batch 回退。

## MLX 源码仓库

```text
third_party/mlx/
```

当前 fork 分支：

```text
multiplex-qwen38-kernels
```

已提交的第一个优化：

```text
17a2154 Add isolated affine4 cross-row QMV path
```

它在 `mlx/backend/metal/kernels/quantized.h` 中增加了独立的
`multiplex_qmv_fast_crossrow_affine4_g64` 路径，只在窄条件下启用；其他 shape
继续走 MLX 原有 `qmv_fast_impl`。

## 入口与默认 MLX

项目入口默认使用 `.venv` 中的 custom MLX：

```text
./serve.sh
./serve-webui.sh
./try_engine.sh
```

这些脚本可用 `MULTIPLEX_PYTHON=/path/to/python` 显式覆盖。直接调用 Python
时也应使用 `.venv/bin/python`；不要用系统 `python3`，否则可能加载 PyPI 的
旧版 MLX。

## baseline/custom 双版本

始终保留两套本地编译结果：

```text
.build/mlx-baseline/   # 当前修改前的 MLX 源码编译版
.build/mlx-custom/     # 当前修改后的 MLX 源码编译版
```

它们必须基于同一个 MLX upstream commit，唯一差别是本地 kernel 修改。不要用
旧版 PyPI wheel 作为 custom kernel 的最终对照。

重建两套版本：

```bash
./scripts/prepare-mlx-variants.sh
```

运行测试时选择版本：

```bash
PYTHONPATH=.build/mlx-baseline .venv/bin/python <benchmark.py>
PYTHONPATH=.build/mlx-custom .venv/bin/python <benchmark.py>
```

## 正确的移植流程

每次只移植一个 kernel 或一个独立 dispatch：

1. 以当前 Python MLX 源码为基线；
2. 对照 challenge 版本，找出新增的函数和 dispatch；
3. 使用独立的 `multiplex_` 前缀，避免函数重定义；
4. 保留现有 kernel、实例和 fallback；
5. 只在明确匹配的 dtype、bits、group size、shape、batch 条件下启用；
6. 先编译，再做 correctness，再做速度对比；
7. 只有没有回退且输出一致时才提交。

不要使用 Python monkey patch 替换 MLX kernel，也不要把 challenge 的 Swift MLX
文件直接当作 Python MLX 文件使用。

## 当前 Qwen3.8 量化 kernel

Qwen3.8-27B-MTPLX-4bit 的典型参数是：

```text
hidden_size: 5120
intermediate_size: 13824
bits: 4
group_size: 64
```

当前已加入的 cross-row QMV 主要针对多行 affine INT4/group-64 投影。真实模型
单 token decode 的收益很小，重点应在 MTP verify 或 batch 输入上观察。

真实模型路径：

```text
/Users/yang.wang/models/Qwen3.8-27B-MTPLX-4bit
```

## 已知测速记录

合成量化矩阵测试中，custom 版本对部分 M=1/2 投影约快 5%～10%，但 M=4、较大
形状存在波动。真实 Qwen3.8 模型单 token decode 测试约为：

```text
baseline: 20.09 tok/s
custom:   20.25 tok/s
```

约提升 0.8%。因此不能把 microbenchmark 结果直接等同于端到端收益；后续应
重点测 batch 和 MTP verify。
