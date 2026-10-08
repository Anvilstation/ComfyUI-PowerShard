"""LTX-2/2.5: заголовки, геометрия, fp8/упакованные форматы, host-ускорители. Без torch/CUDA."""
import json
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from wan_fixtures import write_safetensors

ROOT = Path(__file__).resolve().parents[1]


def tiny_ltxav_shapes(layers=2, inner=64, cross=64, audio_inner=32, keyframes=False):
    shapes = {"adaln_single.emb.timestep_embedder.linear_1.bias": [inner],
              "adaln_single.linear.weight": [6 * inner, inner],
              "audio_adaln_single.linear.weight": [6 * audio_inner, audio_inner],
              "patchify_proj.weight": [inner, 16], "scale_shift_table": [2, inner],
              "audio_scale_shift_table": [2, audio_inner]}
    for i in range(layers):
        p = f"transformer_blocks.{i}."
        shapes[p + "attn2.to_k.weight"] = [32 * (inner // 32), cross]
        shapes[p + "attn1.to_q.weight"] = [inner, inner]
        shapes[p + "scale_shift_table"] = [6, inner]
        shapes[p + "scale_shift_table_a2v_ca_audio"] = [5, audio_inner]
    if keyframes:
        shapes["keyframes_abs_pos_embedding"] = [1, inner]
    return shapes


def write_ltx(path, prefix="", dtype="BF16", metadata=None, extra=None, **kw):
    tensors = {prefix + k: (dtype, v, None) for k, v in tiny_ltxav_shapes(**kw).items()}
    tensors["vae.decoder.conv.weight"] = ("BF16", [4, 4], None)        # полный checkpoint: не-DiT ключи
    tensors["text_embedding_projection.weight"] = ("BF16", [8, 8], None)
    tensors.update(extra or {})
    return write_safetensors(path, tensors, metadata)


def test_ltxav_geometry_prefixes_and_metadata(tmp_path):
    from powershard.ltx_config import LTXCheckpoint, ltx_geometry, describe_ltx, ltx_unet_config
    for prefix in ("", "diffusion_model.", "model.diffusion_model."):
        meta = {"config": json.dumps({"transformer": {"cross_attention_adaln": True, "num_attention_heads": 2,
                                                      "audio_num_attention_heads": 4, "audio_attention_head_dim": 8}})}
        ckpt = LTXCheckpoint(write_ltx(tmp_path / f"m{len(prefix)}.safetensors", prefix=prefix, metadata=meta))
        config = ckpt.model_config()
        assert ckpt.prefix == prefix
        assert config["image_model"] == "ltxav" and config["num_layers"] == 2
        assert config["attention_head_dim"] == 2 and config["cross_attention_dim"] == 64  # comfy: to_k.shape[0] // 32
        assert config["cross_attention_adaln"] is True and config["use_keyframes_abs_pos_embedding"] is False
        g = ltx_geometry(config)
        assert g["av"] and g["heads"] == 2 and g["audio_heads"] == 4 and g["audio_head_dim"] == 8
        assert describe_ltx(config).endswith("xattn_adaln")
        assert ltx_unet_config(config)["disable_unet_model_creation"]
        assert "vae.decoder.conv.weight" in ckpt.ignored or prefix == ""


def test_ltxv_video_only_and_keyframe_marker(tmp_path):
    from powershard.ltx_config import LTXCheckpoint
    shapes = tiny_ltxav_shapes(keyframes=True)
    tensors = {k: ("F16", v, None) for k, v in shapes.items() if not k.startswith("audio_")}
    config = LTXCheckpoint(write_safetensors(tmp_path / "v.safetensors", tensors)).model_config()
    assert config["image_model"] == "ltxv" and config["use_keyframes_abs_pos_embedding"]


def test_not_ltx_and_packed_formats(tmp_path):
    from powershard.ltx_config import LTXCheckpoint, ltx_storage_ok
    from powershard.config import DistributedConfig
    with pytest.raises(ValueError, match="не LTX"):
        LTXCheckpoint(write_safetensors(tmp_path / "x.safetensors", {"head.modulation": ("F16", [1, 6, 8], None)}))
    packed = write_ltx(tmp_path / "nvfp4.safetensors", extra={"transformer_blocks.0.ff.net.0.proj.weight": ("U8", [64, 32], None)})
    with pytest.raises(ValueError, match="без comfy_quant"):
        LTXCheckpoint(packed).quantization()
    ok = LTXCheckpoint(write_ltx(tmp_path / "bf16.safetensors"))
    assert ltx_storage_ok(ok, DistributedConfig().precision) == "bf16"
    assert ltx_storage_ok(ok, "int8_fp16") == "bf16"   # LTX всегда деквантует; precision не ограничивает


def test_fp8_scaled_and_memory_plan(tmp_path):
    from powershard.ltx_config import LTXCheckpoint, ltx_memory_plan, ltx_fp32_parameter
    extra = {"transformer_blocks.0.attn1.to_q.weight": ("F8_E4M3", [64, 64], None),
             "transformer_blocks.0.attn1.to_q.weight_scale": ("F32", [], None)}
    ckpt = LTXCheckpoint(write_ltx(tmp_path / "fp8.safetensors", extra=extra))
    assert ckpt.storage()["kind"] == "fp8_scaled"
    plan = ltx_memory_plan(ckpt, 6)
    assert plan["world_size"] == 6 and plan["shard_bytes_lower_bound"] * 6 >= plan["converted_storage_bytes"]
    assert ltx_fp32_parameter("transformer_blocks.0.scale_shift_table_a2v_ca_audio")
    assert ltx_fp32_parameter("audio_embeddings_connector.learnable_registers")
    assert not ltx_fp32_parameter("transformer_blocks.0.attn1.to_q.weight")


def test_ltx_options_validation():
    from powershard.ltx_config import LTXOptions
    assert LTXOptions().attention_chunk == 8192
    with pytest.raises(ValueError):
        LTXOptions(mlp_chunk_mode="x")
    with pytest.raises(ValueError):
        LTXOptions(attention_chunk=1)


# ---------------------------------------------------------------- host accelerators
def _fn(module):
    def wrapper(executor, *args, **kwargs):
        return executor(*args, **kwargs)
    wrapper.__module__ = module
    return wrapper


def test_host_safe_wrappers_are_stripped():
    from powershard.accel import is_host_safe, strip_host_safe_wrappers
    easy = _fn("comfy_extras.nodes_easycache")
    windows = _fn("comfy.context_windows")
    foreign = _fn("custom_nodes.teacache.nodes")
    assert is_host_safe(easy) and is_host_safe(windows) and not is_host_safe(foreign)
    wrappers = {"diffusion_model": {"easycache": [easy]}, "outer_sample": {"x": [windows, foreign]}}
    assert strip_host_safe_wrappers(wrappers) == {"outer_sample": {"x": [foreign]}}


class _Sigma:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def max(self):
        return self.value

    def __float__(self):
        return float(self.value)


def test_block_cache_holder_steps_and_reset():
    from powershard.accel import BlockCacheHolder
    holder = BlockCacheHolder(0.1, sigma_start=0.9, sigma_end=0.1, max_skips=2, warmup_steps=1)
    opts = lambda s: {"sigmas": _Sigma(s), "cond_or_uncond": [0, 1]}  # noqa: E731
    first = holder.spec(opts(1.0), None, [(2, 16, 3, 4, 4)])
    assert first["reset"] and not first["active"]          # sigma 1.0 > start
    second = holder.spec(opts(0.8), None, [(2, 16, 3, 4, 4)])
    assert not second["reset"] and second["active"] and second["key"] == first["key"]
    again = holder.spec(opts(1.0), None, [(2, 16, 3, 4, 4)])  # новый sampling pass
    assert again["reset"] and again["key"] != first["key"]
    other = holder.spec(opts(0.7), None, [(2, 16, 3, 4, 4)], extra=("stg",))
    assert other["key"] != holder.spec(opts(0.7), None, [(2, 16, 3, 4, 4)])["key"]


def test_nag_rows_and_riflex_index():
    from powershard.accel import nag_rows, riflex_frequency_index
    assert nag_rows([0, 1], 4) == [0, 1]
    assert nag_rows([1, 0], 4) == [2, 3]
    assert nag_rows([], 3) == [0, 1, 2]
    assert riflex_frequency_index(128, 21) == 4   # Wan head_dim 128, 81 кадр -> 21 latent


def test_nag_combine_numpy_reference():
    """Та же формула, что ChenDarYen/ComfyUI-NAG (проверка на numpy)."""
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(size=(3, 5, 8)), rng.normal(size=(3, 5, 8))
    scale, tau, alpha = 5., 2.5, .25
    guide = pos * scale - neg * (scale - 1)
    npos, ng = np.abs(pos).sum(-1, keepdims=True), np.abs(guide).sum(-1, keepdims=True)
    ratio = ng / npos
    guide = np.where(ratio > tau, guide / (ng + 1e-7) * npos * tau, guide)
    expected = guide * alpha + pos * (1 - alpha)
    assert np.isfinite(expected).all() and (np.abs(expected).sum(-1) / npos[..., 0] <= tau * alpha + (1 - alpha) + 1e-6).all()


def test_partial_softmax_allreduce_equals_full_numpy():
    """Алгоритм v2a: online softmax по локальным ключам rank + MAX/SUM all-reduce = полная attention."""
    rng = np.random.default_rng(1)
    q, k, v = rng.normal(size=(2, 3, 7, 4)), rng.normal(size=(2, 3, 50, 4)), rng.normal(size=(2, 3, 50, 4))
    scale = 4 ** -.5
    s = q @ k.swapaxes(-1, -2) * scale
    p = np.exp(s - s.max(-1, keepdims=True))
    full = (p / p.sum(-1, keepdims=True)) @ v
    parts = []
    bounds = [(0, 9), (9, 18), (18, 27), (27, 36), (36, 50), (50, 50)]   # последний rank без ключей
    for a, b in bounds:
        m = np.full(q.shape[:-1] + (1,), -np.inf)
        l, o = np.zeros_like(m), np.zeros(q.shape)
        for c in range(a, b, 4):
            sc = q @ k[..., c:min(c + 4, b), :].swapaxes(-1, -2) * scale
            new = np.maximum(m, sc.max(-1, keepdims=True))
            pc, corr = np.exp(sc - new), np.exp(m - new)
            l, o, m = l * corr + pc.sum(-1, keepdims=True), o * corr + pc @ v[..., c:min(c + 4, b), :], new
        parts.append((m, l, o))
    top = np.max([m for m, _, _ in parts], axis=0)
    with np.errstate(invalid="ignore"):
        l = sum(np.nan_to_num(l * np.exp(m - top)) for m, l, _ in parts)
        o = sum(np.nan_to_num(o * np.exp(m - top)) for m, _, o in parts)
    np.testing.assert_allclose(o / l, full, rtol=1e-10, atol=1e-12)


def test_ltx_workflows_are_valid_api_graphs():
    for path in sorted((ROOT / "workflows_ltx").glob("*.api.json")):
        graph = json.loads(path.read_text())
        for node_id, node in graph.items():
            assert "class_type" in node and "inputs" in node, (path.name, node_id)
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in graph, (path.name, node_id, value)


def test_ltx_nodes_register_without_torch(monkeypatch):
    fake = types.ModuleType("folder_paths")
    fake.get_filename_list = lambda folder: [f"{folder}_a.safetensors"]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    from powershard.ltx_nodes import LTX_NODE_CLASS_MAPPINGS, LTX_NODE_DISPLAY_NAME_MAPPINGS
    assert set(LTX_NODE_CLASS_MAPPINGS) == set(LTX_NODE_DISPLAY_NAME_MAPPINGS)
    loader = LTX_NODE_CLASS_MAPPINGS["PowerShardLTXLoader"].INPUT_TYPES()
    assert "checkpoints/checkpoints_a.safetensors" in loader["required"]["checkpoint"][0]


# ------------------------------------------------------------- quantized formats
def _quant(conf):
    raw = json.dumps(conf).encode()
    return ("U8", [len(raw)], raw)


def test_comfy_quant_logical_shapes_all_formats(tmp_path):
    from powershard.ltx_config import LTXCheckpoint, ltx_storage_ok, ltx_memory_plan
    from powershard.quant_formats import logical_shape, formats_summary
    base = "transformer_blocks.0."
    extra = {
        base + "attn1.to_q.weight": ("U8", [64, 32], None), base + "attn1.to_q.comfy_quant": _quant({"format": "nvfp4"}),
        base + "attn1.to_q.weight_scale": ("F8_E4M3", [64, 4], None), base + "attn1.to_q.weight_scale_2": ("F32", [], None),
        base + "attn2.to_k.weight": ("I8", [64, 64], None),
        base + "attn2.to_k.comfy_quant": _quant({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}),
        base + "attn2.to_k.weight_scale": ("F32", [64, 1], None),
    }
    ckpt = LTXCheckpoint(write_ltx(tmp_path / "q.safetensors", extra=extra))
    assert ckpt.tensors[base + "attn1.to_q.weight"]["shape"] == [64, 64]          # nvfp4: 2 элемента на байт
    assert ckpt.tensors[base + "attn1.to_q.weight"]["stored_shape"] == [64, 32]
    assert ckpt.model_config()["cross_attention_dim"] == 64                          # int8: форма та же
    assert formats_summary(ckpt.quant_configs) == {"nvfp4": 1, "int8_tensorwise": 1}
    assert ltx_storage_ok(ckpt, "int8_fp16").startswith("comfy_quant:")              # precision не мешает LTX
    assert ltx_memory_plan(ckpt, 2)["converted_storage_bytes"] > 0
    assert logical_shape([8, 6], {"format": "w6a8_int8"}) == [8, 8]
    assert logical_shape([8, 4], {"format": "asym_w4a8_int8"}) == [8, 8]
    assert logical_shape([8, 4], {"format": "convrot_w4a4"}) == [8, 8]
    assert logical_shape([8, 8], {"format": "convrot_w4a4", "linear_dtype": "int8"}) == [8, 8]
    assert logical_shape([8, 8], {"format": "mxfp8"}) == [8, 8]


def test_wan_quantized_checkpoint_int8_path_vs_dequant(tmp_path):
    from wan_fixtures import write_tiny_wan
    from powershard.wan_config import WanCheckpoint
    extra = {"blocks.0.ffn.0.weight": ("U8", [512, 128], None),
             "blocks.0.ffn.0.comfy_quant": _quant({"format": "nvfp4"}),
             "blocks.0.ffn.0.weight_scale": ("F8_E4M3", [512, 16], None),
             "blocks.0.ffn.0.weight_scale_2": ("F32", [], None)}
    ckpt = WanCheckpoint(write_tiny_wan(tmp_path / "w.safetensors", extra=extra))
    assert ckpt.tensors["blocks.0.ffn.0.weight"]["shape"] == [512, 256]
    assert ckpt.model_config()["ffn_dim"] == 512
    assert ckpt.storage()["kind"] == "comfy_quant:nvfp4"
    with pytest.raises(ValueError, match="precision=fp16"):
        ckpt.quantization()                                    # native int8 путь невозможен, деквантование — да


def test_weight_dtype_resolution():
    from powershard.quant_formats import resolve_weight_dtype
    v100 = [{"capability": [7, 0]}] * 6
    rtx5090 = [{"capability": [12, 0]}] * 8
    assert resolve_weight_dtype("auto", v100) == "fp16"
    assert resolve_weight_dtype("auto", rtx5090) == "bf16"
    assert resolve_weight_dtype("auto", v100[:1] + rtx5090) == "fp16"
    assert resolve_weight_dtype("bf16", v100) == "bf16"
    with pytest.raises(ValueError):
        resolve_weight_dtype("fp8", v100)


def test_quant_options_and_nodes(monkeypatch):
    from powershard.quant_formats import validate_quant_choice, WEIGHT_FORMATS, COMPUTE_MODES
    from powershard.wan_config import WanOptions
    from powershard.ltx_config import LTXOptions
    for fmt in WEIGHT_FORMATS:
        for mode in COMPUTE_MODES:
            validate_quant_choice(fmt, mode)
    with pytest.raises(ValueError):
        validate_quant_choice("gguf", "auto")
    with pytest.raises(ValueError):
        validate_quant_choice("int4", "fast")
    assert WanOptions(weight_format="int4", compute="native").to_dict()["weight_format"] == "int4"
    assert LTXOptions(weight_format="nvfp4", compute="dequantize", weight_dtype="bf16").weight_format == "nvfp4"
    with pytest.raises(ValueError):
        WanOptions(weight_format="int2")
    fake = types.ModuleType("folder_paths")
    fake.get_filename_list = lambda folder: [f"{folder}_a.safetensors"]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    from powershard.ltx_nodes import LTX_NODE_CLASS_MAPPINGS
    from powershard.wan_nodes import PowerShardWanOptions
    h3 = LTX_NODE_CLASS_MAPPINGS["PowerShardH3QuantLoader"].INPUT_TYPES()
    flat = {**h3["required"], **h3.get("optional", {})}
    assert list(flat["weight_format"][0]) == list(WEIGHT_FORMATS) and list(flat["compute"][0]) == list(COMPUTE_MODES)
    wan = PowerShardWanOptions.INPUT_TYPES()
    assert {"weight_format", "compute", "weight_dtype"} <= set({**wan["required"], **wan.get("optional", {})})
