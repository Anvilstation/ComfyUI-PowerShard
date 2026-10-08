"""Деквантование comfy_quant весов родным кодом ComfyUI (quant_formats.dequantize_module). Нужны torch и --comfy.

  python -m pytest tests/test_quant_formats_torch.py --comfy /path/to/ComfyUI
"""
import json
import pytest

torch = pytest.importorskip("torch")
from wan_reference import import_comfy  # noqa: E402


@pytest.fixture(scope="module")
def comfy(request):
    path = request.config.getoption("--comfy")
    if path is None:
        pytest.skip("NOT_RUN: передайте --comfy")
    import_comfy(path)
    return path


def save(tmp_path, tensors, conf):
    from safetensors.torch import save_file
    tensors = dict(tensors)
    tensors["m.comfy_quant"] = torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)
    path = tmp_path / f"{conf['format']}.safetensors"
    save_file(tensors, str(path))
    return path


def run(path, conf, shape, dtype=torch.float32):
    from safetensors import safe_open
    from powershard.quant_formats import dequantize_module
    with safe_open(str(path), framework="pt", device="cpu") as f:
        entries = {k: k for k in f.keys() if k.startswith("m.") and k != "m.bias"}
        return dequantize_module(f, entries, "m", conf, shape, torch.device("cpu"), dtype)


def test_int8_tensorwise_rowwise(comfy, tmp_path):
    g = torch.Generator().manual_seed(0)
    q = torch.randint(-127, 128, (8, 16), generator=g, dtype=torch.int8)
    scale = torch.rand((8, 1), generator=g) * .01
    conf = {"format": "int8_tensorwise"}
    out = run(save(tmp_path, {"m.weight": q, "m.weight_scale": scale}, conf), conf, [8, 16])
    torch.testing.assert_close(out, q.float() * scale)


def test_fp8_scaled(comfy, tmp_path):
    w = torch.randn(8, 16)
    scale = w.abs().max() / 448.
    q = (w / scale).to(torch.float8_e4m3fn)
    conf = {"format": "float8_e4m3fn"}
    out = run(save(tmp_path, {"m.weight": q, "m.weight_scale": scale.reshape(())}, conf), conf, [8, 16])
    torch.testing.assert_close(out, q.float() * scale)


@pytest.mark.parametrize("layout,fmt", [("TensorCoreNVFP4Layout", "nvfp4"), ("TensorCoreMXFP8Layout", "mxfp8")])
def test_block_formats_roundtrip(comfy, tmp_path, layout, fmt):
    """Квантуем comfy-kitchen, сохраняем как ComfyUI (weight/weight_scale/weight_scale_2), деквантуем PowerShard."""
    import comfy.quant_ops as q
    if fmt not in q.QUANT_ALGOS:
        pytest.skip(f"{fmt} нет в этой сборке comfy-kitchen")
    w = torch.randn(32, 64)
    try:
        qt = q.QuantizedTensor.from_float(w, layout)
    except Exception as error:  # нет backend для квантования на этой платформе
        pytest.skip(f"quantize {fmt}: {error}")
    params = qt._params
    tensors = {"m.weight": qt._qdata}
    if fmt == "nvfp4":
        tensors["m.weight_scale"] = params.block_scale.view(torch.uint8) if params.block_scale.dtype != torch.uint8 else params.block_scale
        tensors["m.weight_scale_2"] = params.scale.float()
    else:
        tensors["m.weight_scale"] = params.scale.view(torch.uint8)
    conf = {"format": fmt}
    out = run(save(tmp_path, tensors, conf), conf, [32, 64])
    torch.testing.assert_close(out, qt.dequantize().float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("weight_format", ["int8", "fp8", "nvfp4", "mxfp8", "int4"])
def test_quant_linear_pack_unpack_dequantize(comfy, weight_format):
    """QuantLinear: байтовый вектор (как после FSDP all-gather) -> QuantizedTensor -> GEMM == dequant reference."""
    from torch import nn
    from powershard.quant_linear import install_quant_linears, quantized_from_float, pack, TARGETS, QuantLinear
    fmt = TARGETS[weight_format]
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(128, 64)).to(torch.bfloat16)
    tensors = {"0.weight": {"key": "0.weight", "dtype": "BF16", "shape": [64, 128]}}
    try:
        report = install_quant_linears(net, tensors, weight_format, "dequantize", None, torch.bfloat16,
                                       torch.device("cpu"))
    except Exception as error:  # формат/квантование не поддержано этой сборкой comfy-kitchen на CPU
        pytest.skip(f"{fmt}: {error}")
    layer = net[0]
    assert isinstance(layer, QuantLinear) and report["modules"][fmt]["count"] == 1
    weight = torch.randn(64, 128, dtype=torch.bfloat16)
    qt = quantized_from_float(weight, fmt)
    layer.qbytes = nn.Parameter(pack(qt, layer._ps_qplan, "cpu"), requires_grad=False)
    layer.bias = nn.Parameter(torch.zeros(64, dtype=torch.bfloat16), requires_grad=False)
    x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    out = layer(x)
    ref = torch.nn.functional.linear(x, qt.dequantize().to(torch.bfloat16))
    assert out.shape == (2, 3, 64)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
