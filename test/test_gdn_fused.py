"""Fused GDN kernels must be bit-identical to mlx-lm's stock GatedDeltaNet
chain (outputs, conv state and fp32 delta state) at every routed width."""
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_map
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

from multiplex.kernel import gdn_fused

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")


def _mixer():
    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=256, intermediate_size=512,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        rms_norm_eps=1e-6, vocab_size=32, linear_num_value_heads=6,
        linear_num_key_heads=2, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4)
    mx.random.seed(0)
    m = GatedDeltaNet(args)
    m.update(tree_map(lambda p: (mx.random.normal(p.shape) * 0.2).astype(mx.bfloat16),
                      m.parameters()))
    m.A_log = m.A_log.astype(mx.bfloat16)
    m.eval()
    assert gdn_fused.unsupported_reason(m) is None
    return m


def _cache(m, B):
    c = ArraysCache(2)
    c[0] = (mx.random.normal((B, 3, m.conv_dim)) * 2).astype(mx.bfloat16)
    c[1] = mx.random.normal((B, m.num_v_heads, 128, 128)) * 0.1
    return c


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("S", [1, 2, 4, 33])
def test_fused_matches_stock_bitwise(B, S):
    m = _mixer()
    x = (mx.random.normal((B, S, 256)) * 2).astype(mx.bfloat16)
    ref, got = _cache(m, B), ArraysCache(2)
    got.cache = [v + 0 for v in ref.cache]
    assert gdn_fused.applies(x, None, got)
    y_ref = m(x, None, ref)
    y_got = gdn_fused.forward(m, x, got)
    for a, b in [(y_ref, y_got), (ref[0], got[0]), (ref[1], got[1])]:
        assert a.dtype == b.dtype and a.shape == b.shape
        assert mx.array_equal(a, b).item()


def test_routing_keeps_stock_for_masked_and_prefill_shapes():
    m = _mixer()
    c = _cache(m, 1)
    x = mx.zeros((1, 4, 256), dtype=mx.bfloat16)
    assert not gdn_fused.applies(x, mx.ones((1, 4), dtype=mx.bool_), c)
    assert not gdn_fused.applies(mx.zeros((1, gdn_fused.MAX_ROWS + 1, 256)), None, c)
    assert not gdn_fused.applies(x, None, ArraysCache(2))        # first prefill piece
