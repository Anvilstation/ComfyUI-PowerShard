"""Общие helpers torch-тестов Wan: native comfy модели (FP32) и PowerShard-сети (FP16 GEMM)."""
import sys
from pathlib import Path


def import_comfy(path):
    sys.path.insert(0, str(Path(path).resolve()))
    saved = sys.argv
    sys.argv = ["powershard-test", "--cpu", "--disable-dynamic-vram", "--use-pytorch-cross-attention"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.ldm.wan.model as wan_model
    finally:
        sys.argv = saved
    return wan_model


def tiny_config(**overrides):
    config = dict(model_type="t2v", patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=256, ffn_dim=512, freq_dim=16,
                  text_dim=32, out_dim=16, num_heads=2, num_layers=2, window_size=(-1, -1), qk_norm=True,
                  cross_attn_norm=True, eps=1e-6)
    config.update(overrides)
    return config


def randomize(net, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + .1 * torch.randn(p.shape, generator=g))
            elif name.endswith("modulation"):
                p.copy_(.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(.05 * torch.randn(p.shape, generator=g))
        for name, module in net.named_modules():
            if type(module).__name__ == "Blur":
                k = torch.tensor([1., 3., 3., 1.])
                kernel = k[None, :] * k[:, None]
                module.kernel.copy_(kernel / kernel.sum())
    return net.eval().requires_grad_(False)


def native_model(wan_model, config, seed=0):
    """Родной класс (WanModel/Vace/S2V/Animate/Camera) в FP32 с comfy ops."""
    import torch
    import comfy.ops
    from powershard.wan_model import model_class, constructor_kwargs
    cls = model_class(config)
    net = cls(**constructor_kwargs(cls, config), dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    return randomize(net, seed)


def _copy_into(meta_net, native):
    import torch
    meta_net.to_empty(device="cpu")
    state = native.state_dict()
    with torch.no_grad():
        for name, p in meta_net.named_parameters():
            p.copy_(state[name].to(p.dtype))
        for name, b in meta_net.named_buffers():
            b.copy_(state[name].to(b.dtype))
    return meta_net.eval().requires_grad_(False)


def powershard_model(wan_model, config, native, options=None, distributed_config=None):
    """Та же подготовка, что WanBackend, но без FSDP/dist: проверка численной эквивалентности."""
    from powershard.attention_policy import AttentionDispatcher
    from powershard.config import DistributedConfig
    from powershard.wan_config import WanOptions
    from powershard.wan_model import (WanEntrypoint, build_network, make_fp32_parameters, configure_network,
                                      qk_norm_names, apply_qk_scales)
    import torch
    options = options or WanOptions()
    dconfig = distributed_config or DistributedConfig(attention_backend="sdpa", sequence_mode="ulysses")
    net = build_network(config)
    with torch.device("meta"):
        make_fp32_parameters(net)
    net = _copy_into(net, native)
    tracker, _ = configure_network(net, dconfig, options, AttentionDispatcher(dconfig))
    names = qk_norm_names(net)
    params = dict(net.named_parameters())
    apply_qk_scales(net, {n: float(params[n].abs().max()) for n in names})
    return WanEntrypoint(net), tracker


def uni3c_pair(main_dim, seed=7, layers=2):
    """Native WanUni3CControlnet (FP32) и PowerShard runtime (root + state) с теми же весами."""
    import types
    import torch
    import comfy.ops
    import comfy.ldm.wan.uni3c as uni3c
    from powershard.attention_policy import AttentionDispatcher
    from powershard.config import DistributedConfig
    from powershard.fp16_safe import FiniteTracker, install_safe_operations
    from powershard.wan_config import WanOptions
    from powershard.wan_model import (WanOperations, Uni3CEntrypoint, install_wan_compute, qk_norm_names,
                                      apply_qk_scales)
    geometry = dict(in_channels=36, conv_out_dim=main_dim, dim=128, ffn_dim=256, num_layers=layers,
                    time_embed_dim=main_dim, out_proj_dim=main_dim, add_channels=7, mid_channels=16)
    native = randomize(uni3c.WanUni3CControlnet(**geometry, device="cpu", dtype=torch.float32,
                                                operations=comfy.ops.disable_weight_init), seed)
    with torch.device("meta"):
        ours = uni3c.WanUni3CControlnet(**geometry, device="meta", dtype=torch.float16, operations=WanOperations)
    ours = _copy_into(ours, native)
    dconfig = DistributedConfig(attention_backend="sdpa")
    options = WanOptions()
    install_safe_operations(ours, FiniteTracker(False), options.fp16_safe)
    state = install_wan_compute(ours, AttentionDispatcher(dconfig), dconfig, options)
    params = dict(ours.named_parameters())
    apply_qk_scales(ours, {n: float(params[n].abs().max()) for n in qk_norm_names(ours)})
    runtime = types.SimpleNamespace(root=Uni3CEntrypoint(ours), state=state)
    return native, runtime


def save_native(native, path, prefix=""):
    from safetensors.torch import save_file
    import torch
    save_file({prefix + k: v.to(torch.float16) if v.is_floating_point() else v for k, v in native.state_dict().items()}, str(path))
    return path


def variant_inputs(case):
    """(config, args, kwargs PowerShard, kwargs native, uni3c?) для варианта; токены режутся по кадрам неровно."""
    import torch
    g = torch.Generator().manual_seed(9)
    rnd = lambda *shape: torch.randn(shape, generator=g)  # noqa: E731
    if case == "i2v":
        config = tiny_config(model_type="i2v", in_dim=36, num_heads=2)
        args = (rnd(2, 36, 3, 4, 6), torch.tensor([[900., 900., 900.], [100., 300., 300.]]), rnd(2, 6, 32))
        kwargs = dict(clip_fea=rnd(2, 4, 1280))
    elif case == "vace":
        config = tiny_config(model_type="vace", vace_layers=1, vace_in_dim=96)
        args = (rnd(2, 16, 3, 4, 6), torch.tensor([900., 300.]), rnd(2, 6, 32))
        kwargs = dict(vace_context=rnd(2, 2, 96, 3, 4, 6), vace_strength=[1.0, .5])
    elif case == "s2v":
        config = tiny_config(model_type="s2v")
        args = (rnd(1, 16, 3, 8, 8), torch.tensor([700.]), rnd(1, 5, 32))
        kwargs = dict(audio_embed=rnd(1, 25, 1024, 12), reference_latent=rnd(1, 16, 1, 8, 8),
                      reference_motion=rnd(1, 16, 19, 8, 8), control_video=rnd(1, 16, 3, 8, 8))
    elif case == "animate":
        config = tiny_config(model_type="animate", in_dim=36, num_layers=5)
        args = (rnd(1, 36, 3, 4, 6), torch.tensor([500.]), rnd(1, 5, 32))
        face = torch.rand((1, 3, 8, 512, 512), generator=g) * 2 - 1
        kwargs = dict(pose_latents=rnd(1, 16, 2, 4, 6), face_pixel_values=face)
        return config, args, kwargs, dict(kwargs, face_pixel_values=face.half().float()), False
    elif case == "camera":
        config = tiny_config(model_type="camera", in_dim=32)
        args = (rnd(1, 32, 3, 4, 6), torch.tensor([600.]), rnd(1, 5, 32))
        kwargs = dict(camera_conditions=rnd(1, 24, 3, 32, 48), clip_fea=rnd(1, 3, 1280))
    elif case == "uni3c":
        config = tiny_config()
        args = (rnd(2, 16, 3, 4, 6), torch.tensor([800., 800.]), rnd(2, 5, 32))
        kwargs = dict(uni3c_render=rnd(1, 16, 3, 4, 6),
                      _powershard_uni3c=dict(path="mem", strength=.8, sigma_start=10., sigma_end=0.))
        return config, args, kwargs, {}, True
    else:
        raise ValueError(case)
    return config, args, kwargs, kwargs, False


def save_uni3c_comfy(native_cnet, path):
    """Сохранить Uni3C с diffusers-именами attention, как файл из models/model_patches."""
    import torch
    from safetensors.torch import save_file
    renames = ((".self_attn.q.", ".self_attn.to_q."), (".self_attn.k.", ".self_attn.to_k."),
               (".self_attn.v.", ".self_attn.to_v."), (".self_attn.o.", ".self_attn.to_out.0."))
    state = {}
    for key, value in native_cnet.state_dict().items():
        for old, new in renames:
            key = key.replace(old, new)
        state[key] = value.to(torch.float16) if value.is_floating_point() else value
    save_file(state, str(path))
    return path


# ------------------------------------------------- HuMo / SCAIL / WanDancer / Animate2 / InfiniteTalk
NEW_CASES = ("humo", "scail", "scail2", "wandancer", "wandancer30", "animate2", "multitalk1", "multitalk2")
ALL_CASES = ("i2v", "vace", "s2v", "animate", "camera", "uni3c") + NEW_CASES


def multitalk_pair(main_dim, layers, out_dim=32, seed=8):
    """Native WanMultiTalkAttentionBlock-и (FP32) и PowerShard runtime с теми же весами."""
    import types
    import torch
    import comfy.ops
    from powershard.attention_policy import AttentionDispatcher
    from powershard.config import DistributedConfig
    from powershard.fp16_safe import FiniteTracker, install_safe_operations
    from powershard.wan_config import WanOptions
    from powershard.wan_model import WanOperations
    from powershard.wan_extras import MultiTalkBlocks, MultiTalkEntrypoint, install_multitalk_compute
    native = randomize(MultiTalkBlocks(main_dim, out_dim, layers, dtype=torch.float32, device="cpu",
                                       operations=comfy.ops.disable_weight_init), seed)
    with torch.device("meta"):
        ours = MultiTalkBlocks(main_dim, out_dim, layers, dtype=torch.float16, device="meta", operations=WanOperations)
    ours = _copy_into(ours, native)
    options = WanOptions()
    install_safe_operations(ours, FiniteTracker(False), options.fp16_safe)
    install_multitalk_compute(ours, AttentionDispatcher(DistributedConfig(attention_backend="sdpa")), options)
    return native, types.SimpleNamespace(root=MultiTalkEntrypoint(ours))


def new_variant_inputs(case):
    """(config, args, kwargs PowerShard, kwargs native, extra) для новых вариантов; extra: multitalk speakers."""
    import torch
    g = torch.Generator().manual_seed(19)
    rnd = lambda *shape: torch.randn(shape, generator=g)  # noqa: E731
    extra = None
    if case == "humo":
        # batch 2 + reference кадр: аудио дополняется нулями, группы токенов = кадры (видео + ref).
        config = tiny_config(model_type="humo")
        args = (rnd(2, 16, 3, 4, 6), torch.tensor([900., 300.]), rnd(2, 6, 32))
        kwargs = dict(audio_embed=rnd(2, 3, 8, 5, 1280) * .1, reference_latent=rnd(2, 16, 1, 4, 6))
    elif case in ("scail", "scail2"):
        config = tiny_config(model_type=case, in_dim=20)
        args = (rnd(1, 20, 3, 4, 6), torch.tensor([650.]), rnd(1, 5, 32))
        # pose в половинном разрешении (RoPE scale 2), референс — 1 кадр перед видео.
        kwargs = dict(pose_latents=rnd(1, 20, 3, 2, 3), reference_latent=rnd(1, 20, 1, 4, 6), clip_fea=rnd(1, 3, 1280))
        if case == "scail2":
            config["mask_in_dim"] = 28
            kwargs.update(ref_mask_latents=rnd(1, 28, 4, 4, 6), sam_latents=rnd(1, 28, 3, 2, 3), ref_mask_flag=False)
    elif case in ("wandancer", "wandancer30"):
        config = tiny_config(model_type="wandancer", in_dim=36, num_layers=5)
        args = (rnd(1, 36, 3, 8, 8), torch.tensor([550.]), rnd(1, 5, 32))
        kwargs = dict(audio_embed=rnd(1, 20, 35), clip_fea=rnd(1, 3, 1280), clip_fea_ref=rnd(1, 2, 1280),
                      fps=24.0 if case == "wandancer" else 30.0, audio_inject_scale=.7)
    elif case == "animate2":
        config = tiny_config(model_type="animate2", in_dim=36)
        args = (rnd(2, 36, 3, 4, 6), torch.tensor([700., 700.]), rnd(2, 5, 32))
        kwargs = dict(pose_latents=rnd(2, 16, 2, 4, 6), clip_fea=rnd(2, 3, 1280), clip_fea_pose=rnd(2, 3, 1280),
                      context_pose=rnd(2, 4, 32), pose_strength=.8, reference_strength=1.3)
    elif case in ("multitalk1", "multitalk2"):
        speakers = 1 if case == "multitalk1" else 2
        config = tiny_config(model_type="i2v", in_dim=36, dim=320, num_heads=2, ffn_dim=640)
        args = (rnd(1, 36, 3, 4, 6), torch.tensor([800.]), rnd(1, 5, 32))
        kwargs = dict(clip_fea=rnd(1, 3, 1280))
        masks = torch.zeros(2, 2 * 3, dtype=torch.bool)
        masks[0, :3], masks[1, 3:] = True, True
        extra = dict(speakers=speakers, audio=rnd(1, 3, 32 * speakers, 32), masks=masks if speakers == 2 else None,
                     scale=.9)
    else:
        raise ValueError(case)
    return config, args, kwargs, kwargs, extra


def our_options(native_options):
    """transformer_options для PowerShard: всё, кроме native patches/audio_embeds (они переданы kwargs)."""
    return {k: v for k, v in native_options.items() if k not in ("patches", "audio_embeds")}


def variant_call(case):
    """Единый вход для тестов/probe: config и builder(native) -> (args, ours_kwargs, native_kwargs, native_options)."""
    import types
    import torch
    if case not in NEW_CASES:
        config, args, kwargs, native_kwargs, is_uni3c = variant_inputs(case)

        def build(native):
            if not is_uni3c:
                return args, kwargs, native_kwargs, {}
            from comfy_extras.nodes_model_patch import WanUni3CCnetPatch
            native_cnet, runtime = uni3c_pair(config["dim"])
            patch = WanUni3CCnetPatch(types.SimpleNamespace(model=native_cnet), None, None, None, .8, 10., 0.)
            patch.prepared_render = kwargs["uni3c_render"]
            options = {"sigmas": torch.tensor([.8]), "cond_or_uncond": [0, 1]}
            return args, dict(kwargs, _powershard_uni3c_runtime=runtime), native_kwargs, \
                dict(options, patches={"double_block": [patch]})
        return config, build
    config, args, kwargs, native_kwargs, extra = new_variant_inputs(case)

    def build(native):
        if extra is None:
            return args, kwargs, native_kwargs, {}
        from comfy.ldm.wan.model_multitalk import MultiTalkCrossAttnPatch, MultiTalkGetAttnMapPatch
        native_blocks, runtime = multitalk_pair(config["dim"], config["num_layers"])
        patches = {"attn2_patch": [MultiTalkCrossAttnPatch(types.SimpleNamespace(model=native_blocks), extra["scale"])]}
        if extra["masks"] is not None:
            patches["attn1_patch"] = [MultiTalkGetAttnMapPatch(extra["masks"])]
        ours = dict(kwargs, _powershard_multitalk=dict(path="mem", audio_scale=extra["scale"]),
                    multitalk_audio=extra["audio"], _powershard_multitalk_runtime=runtime)
        if extra["masks"] is not None:
            ours["multitalk_masks"] = extra["masks"].float()
        return args, ours, native_kwargs, {"patches": patches, "audio_embeds": extra["audio"]}
    return config, build
