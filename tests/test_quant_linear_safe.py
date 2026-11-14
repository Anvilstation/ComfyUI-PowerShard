"""Small CPU regressions for issue #1; no ComfyUI, checkpoint, CUDA or pytest needed.

Run directly: python tests/test_quant_linear_safe.py
The quantized-weight reader is replaced with an in-memory dequantization stub;
the actual QuantLinear, safe GEMM and chunked MLP code run on CPU tensors.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F

from powershard import quant_linear
from powershard.fp16_safe import FiniteTracker, chunked_mlp_forward


def layer(weight, bias=None, *, safe=True, native=False, follow_input=False):
    src = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device="meta")
    src._ps_safe, src._ps_fp32 = safe, False
    plan = SimpleNamespace(total=16, fmt="int8_tensorwise", values=weight)
    result = quant_linear.QuantLinear(src, plan, torch.float16, native, follow_input)
    result.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
    return result


def unpack_stub(full, plan, dtype, with_extras=False):
    # Real QuantizedTensor dequantization honors the orig_dtype supplied to unpack.
    qt = SimpleNamespace(dequantize=lambda: plan.values.to(dtype))
    return qt, getattr(plan, "extras", {})


class QuantLinearSafeTests(unittest.TestCase):
    def setUp(self):
        self.reader = patch.object(quant_linear, "unpack", side_effect=unpack_stub)
        self.reader.start()
        self.addCleanup(self.reader.stop)

    def test_fp32_residual_is_scaled_before_half_cast(self):
        weight = torch.tensor([[1., -0.5], [0.25, 1.]])
        x = torch.tensor([[[131072., 65536.], [-262144., 131072.]]])
        for native in (False, True):
            with self.subTest(native=native), patch.object(quant_linear, "native_linear") as kernel:
                out = layer(weight, native=native)(x)
                self.assertTrue(torch.isfinite(out).all().item())
                self.assertEqual(out.dtype, torch.float32)
                torch.testing.assert_close(out, F.linear(x, weight), rtol=1e-3, atol=1e-3)
                kernel.assert_not_called()

    def test_dequantized_weight_and_bias_are_not_narrowed_before_scaling(self):
        weight = torch.tensor([[131072., 65536.]])
        bias = torch.tensor([131072.])
        x = torch.tensor([0.25, 0.5])
        out = layer(weight, bias)(x)
        self.assertTrue(torch.isfinite(out).all().item())
        torch.testing.assert_close(out, F.linear(x, weight, bias), rtol=1e-3, atol=1e-3)

    def test_pre_quant_scale_is_applied_in_fp32(self):
        weight = torch.tensor([[1., 0.5]])
        x = torch.tensor([[2., 4.]])
        model = layer(weight)
        pre = torch.tensor([131072., 65536.])
        model._ps_qplan.extras = {"pre_quant_scale": pre}
        out = model(x)
        self.assertTrue(torch.isfinite(out).all().item())
        torch.testing.assert_close(out, F.linear(x * pre, weight), rtol=1e-3, atol=1e-3)

    def test_ordinary_and_follow_input_paths_keep_their_dtype(self):
        weight = torch.tensor([[1., -0.5]])
        x = torch.tensor([[2., 4.]])
        for follow_input, dtype in ((False, torch.float16), (True, torch.float32)):
            with self.subTest(follow_input=follow_input):
                out = layer(weight, safe=False, follow_input=follow_input)(x)
                self.assertEqual(out.dtype, dtype)
                torch.testing.assert_close(out, F.linear(x.to(dtype), weight.to(dtype)))

    def test_native_path_remains_available_without_safe_mode(self):
        weight = torch.tensor([[1., 0.5]])
        x = torch.tensor([[2., 4.]])
        expected = F.linear(x.half(), weight.half())
        with patch.object(quant_linear, "native_linear", return_value=expected) as kernel:
            out = layer(weight, safe=False, native=True)(x)
        kernel.assert_called_once()
        torch.testing.assert_close(out, expected)

    def test_chunked_quantized_mlp_matches_unchunked(self):
        fc1 = layer(torch.tensor([[1., 0.], [0., 1.], [0.5, 0.], [0., 0.5]]))
        fc2 = layer(torch.tensor([[1., 0.], [0., 1.]]))
        tracker = FiniteTracker()
        tracker.begin(torch.device("cpu"))
        mlp = SimpleNamespace(fc1=fc1, fc2=fc2, _ps_tracker=tracker, _ps_finite_slot=0,
                              _ps_mlp_chunk=1, _ps_mlp_policy=SimpleNamespace(mlp_chunk_mode="manual"))
        x = torch.tensor([[[512., 1024.], [1024., 512.], [256., 512.]]])
        out = chunked_mlp_forward(mlp, x)
        gate, up = fc1(x).chunk(2, -1)
        expected = fc2(F.silu(gate) * up)
        tracker.finish(out)
        self.assertGreater(mlp._ps_mlp_report["chunks"], 1)
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
