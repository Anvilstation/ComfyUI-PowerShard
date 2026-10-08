"""Родной comfy LTXAV / LTXV BaseModel на host + proxy без весов генератора.

Всё, что ComfyUI делает вокруг модели, остаётся родным и исполняется на host: extra_conds
(guides/keyframes/IC-LoRA, ref_audio, generated keyframes), process_timestep по denoise-маскам,
упаковка аудио+видео latents, context windows (resize_cond_for_context_window), post-CFG
узлы (STG, Modality Guidance, ID-LoRA ref audio), Dual CFG guider, EasyCache/LazyCache.
В workers уходят только тензоры и флаги прохода (run_vx/run_ax, a2v/v2a, STG-блоки).
"""
from pathlib import Path
import torch
import comfy.model_base
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
from .accel import (HOST_ONLY_TRANSFORMER_KEYS, check_easycache_holder, host_safe_model_options, run_diffusion_wrappers,
                    strip_host_safe_wrappers)
from .comfy_adapter import PowerShardPatcher
from .conditioning_cache import content_hash
from .host_guard import strip_internal_wrappers
from .wire import StagedTensor, has_effect, validate_options

PASS_FLAGS = ("run_vx", "run_ax", "a2v_cross_attn", "v2a_cross_attn")
FORWARD_CONDITIONING = ("denoise_mask", "audio_denoise_mask", "guide_attention_entries", "ref_audio", "generated_keyframes")
STAGE_BYTES = 1 << 20


def worker_options(options, allow_host_wrappers, label="LTX"):
    """transformer_options -> (опции для workers, флаги прохода). Host-only ключи и wrappers снимаются."""
    options = dict(options or {})
    flags = {k: bool(options.pop(k)) for k in PASS_FLAGS if k in options}
    stg = options.pop("stg_self_attn_blocks", None)
    if stg:
        flags["stg_blocks"] = sorted(int(i) for i in stg)
    replace = (options.get("patches_replace") or {})
    if any(replace.values()):
        raise ValueError(f"PowerShard {label}: patches_replace ({sorted(replace)}) — сторонний патч блоков не переносится "
                         "в workers. Для кэша шагов используйте 'PowerShard: Block Cache' / EasyCache.")
    options.pop("patches_replace", None)
    for key in HOST_ONLY_TRANSFORMER_KEYS:
        options.pop(key, None)
    callbacks = options.pop("callbacks", None)
    if has_effect(callbacks) and not allow_host_wrappers:
        raise ValueError("Сторонние callbacks в transformer_options: включите allow_host_wrappers в PowerShardConfigTuning")
    wrappers = options.pop("wrappers", None)
    if wrappers:
        rest = strip_host_safe_wrappers(strip_internal_wrappers(wrappers))
        if any(v for groups in rest.values() for v in groups.values()) and not allow_host_wrappers:
            raise ValueError("Сторонние wrappers: установите allow_host_wrappers=True в PowerShard-конфиге "
                             "(допущены без этого: EasyCache, LazyCache, Context Windows)")
    validate_options(options)
    return options, flags


class LTXDiffusionProxy(torch.nn.Module):
    def __init__(self, session, config):
        super().__init__()
        from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, AudioPatchifier
        self.session = session
        self.dtype = torch.float16
        self.config = dict(config)
        self.av = config["image_model"] == "ltxav"
        self.patchifier = SymmetricPatchifier(1, start_end=True)
        if self.av:
            self.a_patchifier = AudioPatchifier(1, start_end=True)
        self.cross_attention_dim = int(config["cross_attention_dim"])
        self.audio_cross_attention_dim = int(config.get("audio_cross_attention_dim", 2048))
        self.caption_channels = int(config.get("caption_channels", 3840))
        self.num_attention_heads = int(config.get("num_attention_heads", 32))
        self.attention_head_dim = int(config["attention_head_dim"])
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.in_channels = int(config.get("in_channels", 128))
        self.num_layers = int(config["num_layers"])
        # Читаются native LTXAV.resize_cond_for_context_window (context windows + guides).
        self.vae_scale_factors = tuple(config.get("vae_scale_factors", (8, 32, 32)))
        self.causal_temporal_positioning = bool(config.get("causal_temporal_positioning", False))

    def _cancel(self):
        return comfy.model_management.throw_exception_if_processing_interrupted

    def _stage(self, tensor, prefix):
        cpu = tensor.detach().to("cpu")
        key = prefix + content_hash(cpu)
        self.session.stage_tensors({key: cpu})
        return StagedTensor(key, index=0)

    def preprocess_text_embeds(self, context, unprocessed=False):
        """Коннекторы текста считаются в workers; результат кэшируется на host (общий conditioning cache)."""
        from .wan_text import shared_cache
        if not unprocessed and context.shape[-1] in (self.cross_attention_dim + self.audio_cross_attention_dim,
                                                     self.caption_channels * 2):
            return context
        cpu = context.detach().float().cpu()
        stamp = Path(self.session.checkpoint).stat()
        key = content_hash(dict(context=cpu, unprocessed=bool(unprocessed), checkpoint=(self.session.checkpoint, stamp.st_size,
                                stamp.st_mtime_ns), loras=self.session.role_options.get("loras"), role="ltx_preprocess"))
        cache = shared_cache()
        result = cache.get(key)
        if result is None:
            result = self.session.call("preprocess", (cpu,), {"unprocessed": bool(unprocessed)}, cancel=self._cancel())
            cache.put(key, result)
        return result.to(device=context.device, dtype=context.dtype)

    def forward(self, x, timestep, context, attention_mask=None, frame_rate=25, transformer_options=None,
                keyframe_idxs=None, control=None, **kwargs):
        if control is not None:
            raise ValueError("ControlNet (cond control) для LTX не поддерживается; используйте IC-LoRA guides (LTXVAddGuide)")
        options = dict(transformer_options or {})
        return run_diffusion_wrappers(self._remote, self, (x, timestep, context),
                                      dict(attention_mask=attention_mask, frame_rate=frame_rate, transformer_options=options,
                                           keyframe_idxs=keyframe_idxs, **kwargs), options)

    def _remote(self, x, timestep, context, attention_mask=None, frame_rate=25, transformer_options=None,
                keyframe_idxs=None, **kwargs):
        full = dict(transformer_options or {})
        options, flags = worker_options(full, self.session.config.allow_host_wrappers)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        unknown = sorted(set(kwargs) - set(FORWARD_CONDITIONING))
        if unknown:
            raise ValueError(f"PowerShard LTX: conditioning {unknown} не поддерживается")
        kwargs.pop("audio_denoise_mask", None)          # уже учтена host process_timestep
        if keyframe_idxs is None or keyframe_idxs.shape[2] == 0:
            kwargs.pop("denoise_mask", None)            # нужна worker-у только для grid mask guides
        staged = self._stage(context.float(), "context_")
        for key, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.numel() * value.element_size() >= STAGE_BYTES:
                kwargs[key] = self._stage(value, key + "_")
        rate = float(frame_rate.reshape(-1)[0]) if isinstance(frame_rate, torch.Tensor) else float(frame_rate)
        extra = dict(kwargs, attention_mask=attention_mask, frame_rate=rate, transformer_options=options,
                     keyframe_idxs=keyframe_idxs, _powershard_flags=flags)
        video = x[0] if isinstance(x, (list, tuple)) else x
        shapes = [tuple(t.shape) for t in (x if isinstance(x, (list, tuple)) else [x])]
        cache = full.get("powershard_block_cache")
        if cache is not None:
            # LTX timestep = sigma (без *1000): текущая sigma берётся из transformer_options["sigmas"].
            extra["_powershard_block_cache"] = cache.spec(full, None, shapes, extra=(sorted(flags.items()),))
        nag = full.get("powershard_nag")
        if nag is not None and nag.active(full, None):
            negative = self.preprocess_text_embeds(nag.context.to(video.device, torch.float32),
                                                   unprocessed=nag.unprocessed)
            extra["_powershard_nag"] = nag.params()
            extra["nag_context"] = self._stage(negative[:1].float(), "nag_context_")
        if isinstance(timestep, (list, tuple)):
            timestep = tuple(t.float() for t in timestep)
        else:
            timestep = timestep.float()
        result = self.session.call("forward", (x, timestep, staged), extra, cancel=self._cancel())
        device = video.device
        return result.to(device) if isinstance(result, torch.Tensor) else [t.to(device) for t in result]


_REMOTE_CLASSES = {}


def remote_class(base):
    if base not in _REMOTE_CLASSES:
        class RemoteLTX(base):
            def get_dtype_inference(self):
                return torch.float32

            def memory_required(self, input_shape, cond_shapes={}):
                import math
                return 8 * 4 * math.prod(input_shape)

            def extra_conds(self, **kwargs):
                for key in ("control", "hooks", "gligen", "additional_models"):
                    if kwargs.get(key) is not None:
                        raise ValueError(f"PowerShard LTX не поддерживает conditioning {key}")
                return super().extra_conds(**kwargs)
        RemoteLTX.__name__ = RemoteLTX.__qualname__ = "PowerShardRemote" + base.__name__
        _REMOTE_CLASSES[base] = RemoteLTX
    return _REMOTE_CLASSES[base]


def validate_host_options(patcher, label):
    """Общая проверка patcher для распределённых генераторов (LTX, Wan): host-only ускорители допустимы."""
    if patcher.patches or patcher.hook_patches or patcher.weight_wrapper_patches or patcher.injections:
        raise ValueError(f"LoRA/hooks/weight patches через native nodes не поддерживаются; используйте PowerShard {label} LoRA")
    if patcher.forced_hooks is not None or patcher.additional_models:
        raise ValueError("Дополнительные модели/hooks не поддерживаются")
    allow = patcher.session.config.allow_host_wrappers
    if not allow:
        wrappers = strip_host_safe_wrappers(strip_internal_wrappers(patcher.wrappers))
        if any(v for groups in wrappers.values() for v in groups.values()) or \
                any(v for groups in patcher.callbacks.values() for v in groups.values()):
            raise ValueError("Сторонние callbacks/wrappers: установите allow_host_wrappers=True в PowerShard-конфиге "
                             "(без этого допущены EasyCache, LazyCache, Context Windows)")
    extra_objects = set(patcher.object_patches) - {"model_sampling"}
    if extra_objects:
        raise ValueError(f"Object patches {sorted(extra_objects)} не переносятся в workers (TeaCache/MagCache/SageAttention-"
                         "патчи forward). Используйте 'PowerShard: Block Cache' и attention backend в PowerShardConfig.")
    rest = {k: v for k, v in host_safe_model_options(patcher.model_options).items()
            if k not in {"transformer_options", "to_load_options"}}
    if any(has_effect(v) for v in rest.values()):
        raise ValueError(f"Сторонние model_options {sorted(k for k, v in rest.items() if has_effect(v))} не поддерживаются")
    to_load = dict(patcher.model_options.get("to_load_options", {}))
    if "wrappers" in to_load:
        to_load["wrappers"] = strip_internal_wrappers(to_load["wrappers"])
    if has_effect(to_load):
        raise ValueError("Сторонние to_load_options не поддерживаются")
    options = patcher.model_options.get("transformer_options", {})
    if options.get("easycache") is not None:
        check_easycache_holder(options["easycache"])
    return options


class LTXPatcher(PowerShardPatcher):
    def clone(self, *args, **kwargs):
        result = comfy.model_patcher.ModelPatcher.clone(self, *args, **kwargs)
        if hasattr(self, "powershard_memory_plan"):
            result.powershard_memory_plan = dict(self.powershard_memory_plan)
        return result

    def with_h3_patch(self, patch):
        raise ValueError("H3 patch nodes не применяются к LTX; используйте PowerShard LTX Options в loader")

    def with_spectrum(self, config):
        raise ValueError("Spectrum реализован только для MiniMax H3; для LTX — 'PowerShard: Block Cache'")

    def add_patches(self, patches, *args, **kwargs):
        if patches:
            raise ValueError("Native LoRA loader не применим к FSDP LTX: подключите 'PowerShard LTX LoRA' ко входу lora "
                             "LTX loader — LoRA (в т.ч. IC-LoRA, distilled) сливается в локальные shards при загрузке")
        return []

    def validate(self):
        options = validate_host_options(self, "LTX")
        worker_options(options, self.session.config.allow_host_wrappers)

    def cleanup(self):
        comfy.model_patcher.ModelPatcher.cleanup(self)
        session = self.session
        if session.config.release_after_sampling and not session.draining:
            session.close()


def host_model(config, session):
    from .ltx_config import ltx_unet_config
    unet_config = ltx_unet_config(config)
    model_config = comfy.model_detection.model_config_from_unet_config(unet_config)
    if model_config is None:
        raise ValueError(f"ComfyUI не распознал LTX конфигурацию: {unet_config}")
    model_config.manual_cast_dtype = torch.float16
    native = model_config.get_model({}, device=torch.device("cpu"))
    native.__class__ = remote_class(type(native))
    native.diffusion_model = LTXDiffusionProxy(session, config)
    native.eval().requires_grad_(False)
    return native


def load_ltx(path, config, options, loras=(), report_dir=None):
    from .devices import resolve_gpu_selection
    from .ltx_config import LTXCheckpoint, describe_ltx, ltx_memory_plan, ltx_storage_ok
    from .patch_config import H3PatchConfig
    from .wan_lora import plan_lora, safetensors_header
    from .wan_runtime import reusable_wan_session
    from dataclasses import replace
    from .quant_formats import resolve_weight_dtype
    ckpt = LTXCheckpoint(path)
    ltx_storage_ok(ckpt, config.precision)
    geometry = ckpt.model_config()
    weight_dtype = resolve_weight_dtype(options.weight_dtype, resolve_gpu_selection(config.gpu_ids))
    options = replace(options, weight_dtype=weight_dtype, fp16_safe=options.fp16_safe and weight_dtype == "fp16")
    plan = [l.stamp() for l in loras if l.strength != 0]
    for item in plan:
        plan_lora(safetensors_header(item["path"]), ckpt.tensors, label=f"LTX: {Path(item['path']).name}")
    comfy_path = Path(comfy.model_base.__file__).resolve().parents[1]
    role_options = dict(kind="ltx", experts={"main": str(ckpt.path)}, loras={"main": plan} if plan else {},
                        options=options.to_dict(), families={"main": describe_ltx(geometry)})
    session = reusable_wan_session(str(ckpt.path), config, comfy_path, report_dir,
                                   patch=H3PatchConfig(enabled=True, fp16_safe=options.fp16_safe,
                                                       debug_finite=options.debug_finite,
                                                       mlp_chunk_tokens=options.mlp_chunk_tokens,
                                                       mlp_chunk_mode=options.mlp_chunk_mode),
                                   role_options=role_options)
    model = host_model(geometry, session)
    selected = resolve_gpu_selection(config.gpu_ids)
    memory = ltx_memory_plan(ckpt, len(selected))
    size = memory["shard_bytes_lower_bound"] if config.weight_placement == "gpu" else 0
    size += (1 + config.prefetch_blocks) * memory["largest_group_bytes_upper_bound"]
    patcher = LTXPatcher(model, torch.device("cuda", int(selected[0]["user_id"])), torch.device("cpu"), size=size)
    patcher.powershard_memory_plan = dict(memory, host_gpu_parameter_budget_bytes=size, family=describe_ltx(geometry),
                                          geometry=geometry,
                                          weight_placement=config.weight_placement)
    return patcher
