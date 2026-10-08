"""Численная эквивалентность PowerShard LTX forward и native comfy LTXAVModel/LTXVModel (CPU, без FSDP).

Запуск: python -m pytest tests/test_ltx_native.py --comfy /path/to/ComfyUI
Native — FP32 с comfy.model_management.in_training=True (чистые torch-пути вместо comfy-kitchen),
PowerShard — FP16 GEMM/attention, FP32 residual -> допуски FP16.
"""
import pytest

torch = pytest.importorskip("torch")
from wan_reference import import_comfy, randomize, _copy_into  # noqa: E402


@pytest.fixture(scope="module")
def comfy_ltx(request):
    path = request.config.getoption("--comfy")
    if path is None:
        pytest.skip("NOT_RUN: передайте --comfy для native LTX tests")
    import_comfy(path)
    import comfy.model_management
    import comfy.ldm.lightricks.av_model as av
    comfy.model_management.in_training = True
    return av


def tiny_config(**overrides):
    config = dict(image_model="ltxav", in_channels=16, audio_in_channels=128, num_layers=2, attention_head_dim=16,
                  num_attention_heads=4, cross_attention_dim=64, audio_attention_head_dim=8, audio_num_attention_heads=4,
                  audio_cross_attention_dim=32, caption_channels=48, connector_attention_head_dim=16,
                  connector_num_attention_heads=3, connector_num_layers=1, rope_type="split",
                  use_keyframes_abs_pos_embedding=False)
    config.update(overrides)
    return config


def pair(config, options=None):
    import comfy.ops
    from comfy.ldm.lightricks.av_model import LTXAVModel
    from comfy.ldm.lightricks.model import LTXVModel
    from powershard.attention_policy import AttentionDispatcher
    from powershard.config import DistributedConfig
    from powershard.ltx_config import LTXOptions
    from powershard.ltx_model import (LTXEntrypoint, build_ltx_network, make_fp32_parameters, configure_ltx,
                                      qk_norm_names, apply_qk_scales)
    cls = LTXAVModel if config["image_model"] == "ltxav" else LTXVModel
    native = cls(**config, dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    native = randomize(native, 3)
    with torch.no_grad():
        for name, p in native.named_parameters():
            if "scale_shift_table" in name or "learnable_registers" in name:
                p.copy_(.1 * torch.randn(p.shape, generator=torch.Generator().manual_seed(len(name))))
    net = build_ltx_network(config)
    with torch.device("meta"):
        make_fp32_parameters(net)
    net = _copy_into(net, native)
    dconfig = DistributedConfig(attention_backend="sdpa", sequence_mode="ulysses")
    tracker, _ = configure_ltx(net, config, options or LTXOptions(mlp_chunk_mode="manual", mlp_chunk_tokens=5),
                               AttentionDispatcher(dconfig), dconfig)
    params = dict(net.named_parameters())
    apply_qk_scales(net, {n: float(params[n].abs().max()) for n in qk_norm_names(net)})
    return native, LTXEntrypoint(net), tracker


def inputs(seed=0, batch=2, frames=3, audio=6, ctx_dim=96):
    g = torch.Generator().manual_seed(seed)
    vx = torch.randn((batch, 16, frames, 4, 4), generator=g)
    ax = torch.randn((batch, 8, audio, 16), generator=g)
    tokens = frames * 16
    vt = torch.full((batch, tokens, 1), .7)
    vt[:, :16] = 0.                                     # I2V: первый кадр уже чистый
    at = torch.tensor([.7] * batch)
    ctx = torch.randn((batch, 7, ctx_dim), generator=g)
    return [vx, ax], (vt, at), ctx


def compare(actual, expected, atol=4e-2):
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.float(), e.float(), atol=atol, rtol=atol)
    else:
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=atol)


def test_ltxav_forward_matches_native(comfy_ltx):
    native, ours, tracker = pair(tiny_config())
    x, t, ctx = inputs()
    with torch.no_grad():
        expected = native([v.clone() for v in x], t, ctx, frame_rate=25, transformer_options={})
        tracker.begin("cpu")
        actual = ours("forward", (x, t, ctx), dict(frame_rate=25, transformer_options={}))
        tracker.finish(actual)
    compare(actual, expected)


def test_ltxav_cross_attention_adaln_and_gated(comfy_ltx):
    config = tiny_config(cross_attention_adaln=True, apply_gated_attention=True, caption_proj_before_connector=True,
                         connector_attention_head_dim=16, connector_num_attention_heads=4,
                         audio_connector_attention_head_dim=8, audio_connector_num_attention_heads=4)
    native, ours, tracker = pair(config)
    x, t, ctx = inputs(1, ctx_dim=64 + 32)
    with torch.no_grad():
        expected = native([v.clone() for v in x], t, ctx, frame_rate=24, transformer_options={})
        actual = ours("forward", (x, t, ctx), dict(frame_rate=24, transformer_options={}))
    compare(actual, expected)


@pytest.mark.parametrize("flags,native_options", [
    ({"stg_blocks": [1]}, {"stg_self_attn_blocks": frozenset({1})}),
    ({"a2v_cross_attn": False, "v2a_cross_attn": False}, {"a2v_cross_attn": False, "v2a_cross_attn": False}),
    ({"run_ax": False}, {"run_ax": False})])
def test_pass_flags_match_native(comfy_ltx, flags, native_options):
    native, ours, _ = pair(tiny_config())
    x, t, ctx = inputs(2)
    with torch.no_grad():
        expected = native([v.clone() for v in x], t, ctx, frame_rate=25, transformer_options=dict(native_options))
        actual = ours("forward", (x, t, ctx), dict(frame_rate=25, transformer_options={}, _powershard_flags=flags))
    compare(actual, expected)


def test_preprocess_text_embeds_matches_native(comfy_ltx):
    native, ours, _ = pair(tiny_config())
    raw = torch.randn((1, 9, 48), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        expected = native.preprocess_text_embeds(raw, unprocessed=True)
        actual = ours("preprocess", (raw,), {"unprocessed": True})
    compare(actual, expected, atol=2e-2)


def test_ltxv_video_only_matches_native(comfy_ltx):
    config = tiny_config(image_model="ltxv")
    for key in [k for k in config if k.startswith(("audio_", "connector_"))]:
        config.pop(key)
    native, ours, _ = pair(config)
    g = torch.Generator().manual_seed(5)
    x = torch.randn((2, 16, 2, 4, 4), generator=g)
    t = torch.tensor([.6, .6])
    ctx = torch.randn((2, 5, 48), generator=g)
    with torch.no_grad():
        expected = native(x, t, ctx, None, frame_rate=25, transformer_options={})
        actual = ours("forward", (x, t, ctx), dict(frame_rate=25, transformer_options={}))
    compare(actual, expected)


def test_batch_chunk_equals_full(comfy_ltx):
    from powershard.ltx_config import LTXOptions
    _, full, _ = pair(tiny_config())
    _, chunked, _ = pair(tiny_config(), LTXOptions(batch_chunk=1))
    x, t, ctx = inputs(6)
    with torch.no_grad():
        a = full("forward", (x, t, ctx), dict(frame_rate=25, transformer_options={}))
        b = chunked("forward", (x, t, ctx), dict(frame_rate=25, transformer_options={}))
    compare(b, a, atol=1e-3)
