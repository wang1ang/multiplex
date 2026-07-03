# multiplex

本地 LLM 推理框架,核心依赖 [mlx-lm](https://github.com/ml-explore/mlx-lm)。

目标是把三件事接到同一条执行路径上:

- 真批量解码:多条序列一次 forward。
- MTP 投机解码:有 MTP sidecar 时保留 speculative speedup。
- 动态批:请求可以在运行中加入/退出,不要求整批同生同死。

当前 Python 包名和项目名都是 `multiplex`。

## 设计原则

- **直筒,不留分支。** 只实现当前要走的路径;不用的 fallback 和兜底不写。
- **要深,不要宽。** 优先把单一路径的正确性、cache 状态和协议边界打透。
- **分层不越界。** L1/L2 不做路由;L5 只做协议翻译;调度和 cache 归 L3/L4。

## 当前分层

| 层 | 模块 | 职责 | 状态 |
|---|---|---|---|
| L1 引擎 | `multiplex.engine` | 批量 `prefill` / `forward`,logits,cache filter/extract/merge,SSM snapshot/restore,attention trim。 | 可用 |
| L2 投机核 | `multiplex.mtp` | 加载 MTP sidecar,批量 draft/verify,按 batch 最小接受数提交;无头模型走纯 AR。 | 可用 |
| L3 调度机制 | `multiplex.scheduler` | 分块 prefill,merge into live batch,decode step,EOS/max-token 退出,取消,前缀 cache 接入。 | 可用,仍在打磨 |
| L4 Hub | `multiplex.hub` | 单 engine 线程持有 MLX,多 HTTP/调用线程提交请求和接收流式 delta。 | 可用 |
| L5 HTTP | `multiplex.server` | OpenAI-compatible `/v1/chat/completions`, `/v1/responses`, `/v1/models`,SSE,tool-call 桥接。 | 可用,兼容性继续补 |
| Bridge | `multiplex.bridge` | 消息归一、thinking/tool-call 文本过滤与结构化解析。 | 可用 |
| Prefix cache | `multiplex.prefixcache` | token 前缀最长匹配、prompt/session 两个 LRU pool、可选磁盘持久化。 | 可用,仍需实测 |
| Registry | `multiplex.registry` | 扫描 `~/.mtplx/models` 并解析 `--model`。 | 可用 |

## 运行

安装依赖:

```bash
pip install -r requirements.txt
# 或按 pyproject:
pip install -e ".[cli]"
```

交互式动态批 REPL:

```bash
python try_engine.py --model /path/to/model
python try_engine.py --model MODEL_NAME -d 0     # 强制纯 AR
```

HTTP 服务:

```bash
python -m multiplex.server --model /path/to/model --host 127.0.0.1 --port 8000
```

可用端点:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

`--model` 可以是本地路径,也可以是 `~/.mtplx/models` 下的目录名。不传时,如果只发现一个模型会自动选中;发现多个模型时,交互式终端会让你选择,非交互环境会要求显式传 `--model`。

MTP sidecar 自动查找:

- `<model>/mtplx_runtime.json` 中的 `mtp_sidecar_file` 或 `mtp_file`
- `<model>/mtp.safetensors`
- `<model>/mtp/weights.safetensors`

找不到 MTP head 时不会报错,调度器会把 draft depth 强制为 `0`,即纯 AR。

## Prefix Cache

L3 会在 chunk 边界捕获 cache 快照,新请求按 token 前缀找最长可复用边界,命中后只 prefill 剩余 tail。

默认 HTTP/Hub 使用:

```text
~/.cache/multiplex/prefixcache/<model-name>-<model-path-sha>
```

可以通过 `--prefix-cache-dir` 指定目录;传空值、`none` 或 `off` 可关闭磁盘目录。内存侧当前按 `prompt` 和 `session` 两个 pool 各自做 LRU。

## 测试和验证

```bash
python smoke_engine.py /path/to/model
python test_scheduler.py /path/to/model
```

`smoke_engine.py` 验证 L1 的 batch 与 next-k forward。

`test_scheduler.py` 是调度正确性测试脚本,但当前有一个已知文档化缺口:脚本末尾仍引用旧的 `Hub.stream_text` API,而当前 Hub 暴露的是 `stream_messages(...)`。在修测试前,优先用 `try_engine.py` 和 HTTP 端到端验证 L3-L5。

本仓库依赖 MLX/Metal;在没有可用 Metal device 的沙箱或 headless 环境中,导入 `mlx` 可能直接失败。

## 当前边界

- 不是 paged KV cache;当前走 padding/mask 和 batch 行增删。
- 没有多用户公平性、抢占或复杂 QoS。
- prefix cache 已有内存/磁盘机制,但命中率、磁盘清理策略和长会话行为仍需要真实负载压测。
- 批量 MTP 因“取最小接受数”会被最差行拖累;小 batch 和相似请求更适合,大 batch 可能应走纯 AR。
