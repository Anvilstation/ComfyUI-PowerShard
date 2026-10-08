"""Численная эквивалентность PowerShard Wan forward и native comfy WanModel (CPU, без FSDP).

Запуск: python -m pytest tests/test_wan_native.py --comfy /path/to/ComfyUI
FP32 native reference против FP16 GEMM/attention PowerShard -> допуски FP16.
"""
import pytest

torch = pytest.importorskip("torch")
from wan_reference import import_comfy, tiny_config, native_model, powershard_model  # noqa: E402


@pytest.fixture(scope="module")
def wan(request):
    path = request.config.getoption("--comfy")
    if path is None:
        pytest.skip("NOT_RUN: передайте --comfy для native Wan tests")
    return import_comfy(path)


def run_both(wan, config, x, t, ctx, options=None, **kwargs):
    native = native_model(wan, config)
    ours, tracker = powershard_model(wan, config, native, options)
    with torch.no_grad():
        expected = native(x, t, ctx, transformer_options={}, **{k: v for k, v in kwargs.items() if not k.startswith("_")})
        tracker.begin("cpu")
        actual = ours("forward", (x, t, ctx), dict(transformer_options={}, **kwargs))
        tracker.finish(actual)
    assert actual.shape == expected.shape and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), atol=3e-2, rtol=3e-2)
    return actual, expected


@pytest.mark.parametrize("safe", [False, True])
def test_t2v_matches_native(wan, safe):
    from powershard.wan_config import WanOptions
    g = torch.Generator().manual_seed(1)
    x = torch.randn((2, 16, 3, 5, 7), generator=g)  # нечётные H/W -> native padding
    t = torch.tensor([900., 300.])
    ctx = torch.randn((2, 9, 32), generator=g)
    run_both(wan, tiny_config(), x, t, ctx, WanOptions(fp16_safe=safe, mlp_chunk_mode="manual", mlp_chunk_tokens=7))


def test_per_frame_timesteps_ti2v_style(wan):
    g = torch.Generator().manual_seed(2)
    config = tiny_config(in_dim=48, out_dim=48)
    x = torch.randn((1, 48, 3, 4, 6), generator=g)
    t = torch.tensor([[0., 500., 500.]])  # WAN22.process_timestep: per latent frame
    run_both(wan, config, x, t, torch.randn((1, 5, 32), generator=g))


def test_i2v_clip_and_time_concat(wan):
    g = torch.Generator().manual_seed(3)
    config = tiny_config(model_type="i2v", in_dim=36)
    x = torch.randn((2, 36, 2, 4, 4), generator=g)
    run_both(wan, config, x, torch.tensor([700., 700.]), torch.randn((2, 6, 32), generator=g),
             clip_fea=torch.randn((2, 5, 1280), generator=g), time_dim_concat=torch.randn((2, 36, 1, 4, 4), generator=g))


def test_reference_latent_and_rope_options(wan):
    g = torch.Generator().manual_seed(4)
    config = tiny_config(in_dim_ref_conv=16)
    native = native_model(wan, config)
    ours, tracker = powershard_model(wan, config, native)
    x = torch.randn((1, 16, 2, 4, 4), generator=g)
    ref = torch.randn((1, 16, 4, 4), generator=g)
    ctx = torch.randn((1, 4, 32), generator=g)
    rope = {"scale_t": 1.5, "shift_y": 2.0}
    with torch.no_grad():
        expected = native(x, torch.tensor([400.]), ctx, transformer_options={"rope_options": rope}, reference_latent=ref)
        actual = ours("forward", (x, torch.tensor([400.]), ctx),
                      dict(transformer_options={}, reference_latent=ref, _powershard_rope_options=rope))
    torch.testing.assert_close(actual, expected.float(), atol=3e-2, rtol=3e-2)


def test_batch_chunk_equals_full_batch(wan):
    from powershard.wan_config import WanOptions
    g = torch.Generator().manual_seed(5)
    config = tiny_config()
    native = native_model(wan, config)
    full, _ = powershard_model(wan, config, native)
    split, _ = powershard_model(wan, config, native, WanOptions(batch_chunk=1))
    x, ctx = torch.randn((2, 16, 2, 4, 4), generator=g), torch.randn((2, 5, 32), generator=g)
    args = (x, torch.tensor([800., 200.]), ctx)
    with torch.no_grad():
        torch.testing.assert_close(split("forward", args, dict(transformer_options={})),
                                   full("forward", args, dict(transformer_options={})), atol=1e-4, rtol=1e-4)


def test_unsupported_conditioning_is_explicit(wan):
    config = tiny_config()
    ours, _ = powershard_model(wan, config, native_model(wan, config))
    with pytest.raises(ValueError, match="context_latents"):
        ours("forward", (torch.zeros(1, 16, 1, 2, 2), torch.tensor([1.]), torch.zeros(1, 2, 32)),
             dict(transformer_options={}, context_latents=[torch.zeros(1, 16, 1, 2, 2)]))


def test_lora_merger_rows_equal_full_delta(tmp_path, wan):
    from safetensors.torch import save_file
    from powershard.wan_lora import LoraMerger
    from wan_fixtures import tiny_wan_shapes
    g = torch.Generator().manual_seed(6)
    up, down = torch.randn((256, 4), generator=g), torch.randn((4, 256), generator=g)
    diff_b = torch.randn((256,), generator=g)
    path = tmp_path / "l.safetensors"
    save_file({"diffusion_model.blocks.0.self_attn.q.lora_up.weight": up.half(),
               "diffusion_model.blocks.0.self_attn.q.lora_down.weight": down.half(),
               "diffusion_model.blocks.0.self_attn.q.alpha": torch.tensor(2.),
               "diffusion_model.blocks.0.self_attn.q.diff_b": diff_b.half()}, str(path))
    merger = LoraMerger([dict(path=str(path), strength=.5, apply_to="all")],
                        {k: {"shape": s} for k, s in tiny_wan_shapes().items()})
    full = .5 * (up.half().float() @ down.half().float()) * (2. / 4)
    pieces = [merger.delta("blocks.0.self_attn.q.weight", a, b, [256, 256]) for a, b in ((0, 100), (100, 256), (256, 256))]
    torch.testing.assert_close(torch.cat(pieces), full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(merger.delta("blocks.0.self_attn.q.bias", 10, 20, [256]), .5 * diff_b.half().float()[10:20])
    assert merger.delta("blocks.0.self_attn.k.weight", 0, 10, [256, 256]) is None


# ------------------------------------------------------------ VACE / S2V / Animate / Camera / Uni3C
def compare(native, ours, tracker, args, native_kwargs, our_kwargs=None, atol=4e-2):
    with torch.no_grad():
        expected = native(*args, transformer_options={}, **native_kwargs)
        tracker.begin("cpu")
        actual = ours("forward", args, dict(transformer_options={}, **(our_kwargs if our_kwargs is not None else native_kwargs)))
        tracker.finish(actual)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected.float(), atol=atol, rtol=atol)


def test_vace_matches_native(wan):
    g = torch.Generator().manual_seed(11)
    config = tiny_config(model_type="vace", vace_layers=1, vace_in_dim=96)
    native = native_model(wan, config)
    ours, tracker = powershard_model(wan, config, native)
    x = torch.randn((2, 16, 2, 4, 6), generator=g)
    vace = torch.randn((2, 2, 96, 2, 4, 6), generator=g)  # два VACE контекста
    compare(native, ours, tracker, (x, torch.tensor([900., 400.]), torch.randn((2, 5, 32), generator=g)),
            dict(vace_context=vace, vace_strength=[1.0, 0.5]))


def test_camera_matches_native(wan):
    g = torch.Generator().manual_seed(12)
    config = tiny_config(model_type="camera", in_dim=32)
    native = native_model(wan, config)
    ours, tracker = powershard_model(wan, config, native)
    x = torch.randn((1, 32, 2, 4, 4), generator=g)
    camera = torch.randn((1, 24, 2, 32, 32), generator=g)
    compare(native, ours, tracker, (x, torch.tensor([600.]), torch.randn((1, 5, 32), generator=g)),
            dict(camera_conditions=camera, clip_fea=torch.randn((1, 3, 1280), generator=g)))


def test_s2v_matches_native(wan):
    g = torch.Generator().manual_seed(13)
    config = tiny_config(model_type="s2v")
    native = native_model(wan, config)
    ours, tracker = powershard_model(wan, config, native)
    frames = 2
    x = torch.randn((1, 16, frames, 8, 8), generator=g)
    kwargs = dict(audio_embed=torch.randn((1, 25, 1024, frames * 4), generator=g),
                  reference_latent=torch.randn((1, 16, 1, 8, 8), generator=g),
                  reference_motion=torch.randn((1, 16, 19, 8, 8), generator=g),
                  control_video=torch.randn((1, 16, frames, 8, 8), generator=g))
    compare(native, ours, tracker, (x, torch.tensor([700.]), torch.randn((1, 5, 32), generator=g)), kwargs)


def test_animate_matches_native(wan):
    g = torch.Generator().manual_seed(14)
    config = tiny_config(model_type="animate", in_dim=36, num_layers=5)
    native = native_model(wan, config)
    ours, tracker = powershard_model(wan, config, native)
    x = torch.randn((1, 36, 3, 4, 4), generator=g)
    kwargs = dict(pose_latents=torch.randn((1, 16, 2, 4, 4), generator=g),
                  face_pixel_values=torch.rand((1, 3, 8, 512, 512), generator=g) * 2 - 1,
                  clip_fea=torch.randn((1, 3, 1280), generator=g))
    # Native считает motion encoder в dtype входа; PowerShard — в FP16 (как native fp16 модель).
    native_kwargs = dict(kwargs, face_pixel_values=kwargs["face_pixel_values"].half().float())
    compare(native, ours, tracker, (x, torch.tensor([500.]), torch.randn((1, 5, 32), generator=g)), native_kwargs, kwargs,
            atol=6e-2)


def test_uni3c_controlnet_matches_native_patch(wan):
    import types
    from comfy_extras.nodes_model_patch import WanUni3CCnetPatch
    from wan_reference import uni3c_pair
    g = torch.Generator().manual_seed(15)
    config = tiny_config()
    native = native_model(wan, config)
    ours, tracker = powershard_model(wan, config, native)
    native_cnet, runtime = uni3c_pair(config["dim"])
    x = torch.randn((2, 16, 2, 4, 6), generator=g)
    render = torch.randn((1, 16, 2, 4, 6), generator=g)
    patch = WanUni3CCnetPatch(types.SimpleNamespace(model=native_cnet), None, None, None, .8, 10., 0.)
    patch.prepared_render = render
    t, ctx = torch.tensor([800., 800.]), torch.randn((2, 5, 32), generator=g)
    options = {"sigmas": torch.tensor([.8]), "cond_or_uncond": [0, 1]}
    with torch.no_grad():
        expected = native(x, t, ctx, transformer_options=dict(options, patches={"double_block": [patch]}))
        tracker.begin("cpu")
        actual = ours("forward", (x, t, ctx), dict(transformer_options=options, uni3c_render=render,
                      _powershard_uni3c=dict(path="mem", strength=.8, sigma_start=10., sigma_end=0.),
                      _powershard_uni3c_runtime=runtime))
        tracker.finish(actual)
        plain = ours("forward", (x, t, ctx), dict(transformer_options=options))
    torch.testing.assert_close(actual, expected.float(), atol=4e-2, rtol=4e-2)
    assert not torch.allclose(actual, plain), "Uni3C residual не применился"


# ---------------------------------------------- HuMo / SCAIL / SCAIL2 / WanDancer / Animate2 / InfiniteTalk
from wan_reference import NEW_CASES  # noqa: E402


@pytest.mark.parametrize("case", NEW_CASES)
def test_new_variants_match_native(wan, case):
    """Native comfy класс варианта (FP32) против PowerShard forward с FP16 GEMM/attention."""
    from wan_reference import variant_call, our_options
    config, build = variant_call(case)
    native = native_model(wan, config)
    ours, tracker = powershard_model(wan, config, native)
    args, kwargs, native_kwargs, native_options = build(native)
    with torch.no_grad():
        expected = native(*args, transformer_options=dict(native_options), **native_kwargs)
        tracker.begin("cpu")
        actual = ours("forward", args, dict(kwargs, transformer_options=our_options(native_options)))
        tracker.finish(actual)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected.float(), atol=4e-2, rtol=4e-2)


def test_animate2_pose_cache_matches_uncached(wan):
    """WanAnimate2Cache-аналог в worker: второй шаг берёт входы pose branch из RAM (kv_from_input)."""
    from wan_reference import new_variant_inputs
    config, args, kwargs, _, _ = new_variant_inputs("animate2")
    args = tuple(a[:1] for a in args)
    kwargs = {k: (v[:1] if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    native = native_model(wan, config)
    ours, tracker = powershard_model(wan, config, native)
    spec = dict(id="cache-1", key="pose-a", dtype="default")
    with torch.no_grad():
        expected = native(*args, transformer_options={}, **kwargs)
        outputs = []
        for _ in range(2):
            tracker.begin("cpu")
            outputs.append(ours("forward", args, dict(kwargs, transformer_options={}, _powershard_animate2_cache=spec)))
            tracker.finish(outputs[-1])
    slot = ours.network._ps_pose_cache.slots["pose-a"]
    assert len(slot["blocks"]) == config["num_layers"]
    for actual in outputs:
        torch.testing.assert_close(actual, expected.float(), atol=5e-2, rtol=5e-2)


def test_wandancer_fused_in_proj_loads_by_row_offset(tmp_path, wan):
    """WanDancer checkpoint с nn.MultiheadAttention in_proj: q/k/v читаются смещением строк."""
    from safetensors.torch import save_file
    from powershard.wan_config import WanCheckpoint
    from wan_reference import new_variant_inputs
    config = new_variant_inputs("wandancer")[0]
    native = native_model(wan, config)
    state = {k: v.half() for k, v in native.state_dict().items()}
    for i in range(2):
        p = f"music_encoder.{i}.self_attn."
        for suffix in ("weight", "bias"):
            state[p + "in_proj_" + suffix] = torch.cat([state.pop(p + f"{x}_proj.{suffix}") for x in "qkv"])
    save_file(state, str(tmp_path / "dancer.safetensors"))
    ckpt = WanCheckpoint(tmp_path / "dancer.safetensors")
    assert ckpt.model_config()["model_type"] == "wandancer"
    for i in range(2):
        for x, offset in zip("qkv", (0, 256, 512)):
            assert ckpt.tensors[f"music_encoder.{i}.self_attn.{x}_proj.weight"]["row_offset"] == offset


def test_i2v_model_without_clip_uses_full_context_image_branch(wan):
    """Native WanI2VCrossAttention при context_img_len=None: k_img/v_img по всему контексту (Animate2/InfiniteTalk без CLIP)."""
    g = torch.Generator().manual_seed(21)
    config = tiny_config(model_type="i2v", in_dim=36)
    run_both(wan, config, torch.randn((1, 36, 2, 4, 6), generator=g), torch.tensor([500.]), torch.randn((1, 5, 32), generator=g))
