"""Родной comfy WAN21/WAN22 BaseModel на host + proxy без весов генератора.

Все выходы MoE loader (high / low / moe) — clones ОДНОГО ModelPatcher с общей
моделью-proxy и общей WanSession. Эксперт выбирается ключом в
model_options["transformer_options"], поэтому ComfyUI видит clones (is_clone),
не выгружает «соседа» при переходе high -> low и не трогает workers.
"""
from pathlib import Path
import types
import torch
import comfy.model_base
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
from .comfy_adapter import PowerShardPatcher
from .conditioning_cache import content_hash
from .host_guard import strip_internal_wrappers
from .wire import StagedTensor, has_effect, validate_options
from .wan_config import WanCheckpoint, comfy_unet_config, describe_family, memory_plan, same_geometry

EXPERT_KEY = "powershard_wan_expert"
BOUNDARY_KEY = "powershard_wan_boundary"
UNI3C_KEY = "powershard_wan_uni3c"
MULTITALK_KEY = "powershard_wan_multitalk"
ANIMATE2_CACHE_KEY = "powershard_wan_animate2_cache"
INTERNAL_KEYS = (EXPERT_KEY, BOUNDARY_KEY, UNI3C_KEY, MULTITALK_KEY, ANIMATE2_CACHE_KEY)
FORWARD_CONDITIONING = ("clip_fea", "time_dim_concat", "reference_latent",
                        "vace_context", "vace_strength",                         # VACE
                        "audio_embed", "reference_motion", "control_video",      # S2V / HuMo
                        "pose_latents", "face_pixel_values",                     # Animate / SCAIL / Animate2
                        "camera_conditions",                                     # Fun Camera
                        "ref_mask_latents", "sam_latents", "ref_mask_flag",      # SCAIL-2
                        "clip_fea_ref", "fps", "audio_inject_scale",             # WanDancer
                        "clip_fea_pose", "context_pose", "pose_strength", "reference_strength")  # Animate2
# Host-only wrappers native InfiniteTalk: подмена первых латентов до apply_model и склейка после
# sampling выполняются на host и в workers не передаются.
HOST_WRAPPERS = {("outer_sample", "infinite_talk_outer_sample"): "InfiniteTalkOuterSampleWrapper",
                 ("apply_model", "MultiTalk_apply_model"): "MultiTalkApplyModelWrapper"}


def strip_wan_host_wrappers(wrappers):
    """Убирает только native wrappers InfiniteTalk (по ключу И классу); остальное идёт в обычную проверку."""
    out = {}
    for kind, keyed in (wrappers or {}).items():
        out[kind] = {}
        for key, functions in keyed.items():
            name = HOST_WRAPPERS.get((kind, key))
            kept = [f for f in functions if not (name and type(f).__name__ == name)]
            if kept:
                out[kind][key] = kept
    return out


def strip_host_callbacks(callbacks):
    """Убирает cleanup-callback native WanAnimate2Cache (освобождает только host PoseBranchCache)."""
    out = {}
    for kind, keyed in (callbacks or {}).items():
        for key, functions in (keyed or {}).items():
            kept = [f for f in functions if getattr(f, "__qualname__", "").split(".")[0] != "WanAnimate2Cache"]
            if kept:
                out.setdefault(kind, {})[key] = kept
    return out


def native_patch_error(patches):
    names = {type(p).__name__ for group in (patches or {}).values() for p in (group if isinstance(group, (list, tuple)) else [])}
    if "WanUni3CCnetPatch" in names:
        return ("Native 'Apply Wan Uni3C ControlNet' не переносится в workers: используйте "
                "'PowerShard Wan: Uni3C ControlNet' с тем же файлом из models/model_patches")
    if names & {"MultiTalkCrossAttnPatch", "MultiTalkGetAttnMapPatch"}:
        return ("Native 'WanInfiniteTalkToVideo' с MODEL_PATCH не переносится в workers: используйте "
                "'PowerShard Wan: InfiniteTalk' (тот же файл из models/model_patches, те же входы)")
    return None
STAGE_BYTES = 1 << 20  # постоянные за прогон большие conditioning пишутся в run-stage один раз


class _WeightShape:
    """WAN21.concat_cond читает только patch_embedding.weight.shape[1]."""
    def __init__(self, shape):
        self.weight = types.SimpleNamespace(shape=torch.Size(shape))


class WanDiffusionProxy(torch.nn.Module):
    def __init__(self, session, geometry):
        super().__init__()
        self.session = session
        self.dtype = torch.float16
        self.geometry = dict(geometry)
        self.patch_size = tuple(geometry["patch_size"])
        self.dim, self.num_heads = geometry["dim"], geometry["num_heads"]
        self.in_dim, self.out_dim = geometry["in_dim"], geometry["out_dim"]
        self.patch_embedding = _WeightShape((geometry["dim"], geometry["in_dim"]) + self.patch_size)

    def _call(self, command, args, kwargs, device):
        result = self.session.call(command, args, kwargs,
                                   cancel=comfy.model_management.throw_exception_if_processing_interrupted)
        return result.to(device) if isinstance(result, torch.Tensor) else [x.to(device) for x in result]

    def _stage_conditioning(self, context, prefix="context_"):
        cpu = context.detach().to("cpu")
        key = prefix + content_hash(cpu)
        self.session.stage_tensors({key: cpu})
        return StagedTensor(key, index=0)

    @staticmethod
    def select_expert(options, timestep):
        slot = options.get(EXPERT_KEY, "main")
        if slot != "moe":
            return slot
        # Wan 2.2: high-noise эксперт пока t >= boundary * 1000 (t = sigma * 1000 у FLOW).
        boundary = float(options.get(BOUNDARY_KEY, .875))
        value = float(timestep.detach().float().max())
        return "high" if value >= boundary * 1000. else "low"

    def forward(self, x, timestep, context, control=None, transformer_options=None, **kwargs):
        # WrappersMP.DIFFUSION_MODEL (EasyCache) исполняются на host вокруг RPC, как в native WanModel.forward.
        from .accel import run_diffusion_wrappers
        options = dict(transformer_options or {})
        return run_diffusion_wrappers(self._remote, self, (x, timestep, context),
                                      dict(control=control, transformer_options=options, **kwargs), options)

    def _remote(self, x, timestep, context, control=None, transformer_options=None, **kwargs):
        from .accel import HOST_ONLY_TRANSFORMER_KEYS, strip_host_safe_wrappers
        if control is not None:
            raise ValueError("Классический ControlNet (cond control) для Wan в ComfyUI нет; используйте "
                             "'PowerShard Wan: Uni3C ControlNet', VACE или Fun Control (concat)")
        full = dict(transformer_options or {})
        options = dict(full)
        for key in HOST_ONLY_TRANSFORMER_KEYS:
            options.pop(key, None)
        error = native_patch_error(options.get("patches"))
        if error:
            raise ValueError(error)
        slot = self.select_expert(options, timestep)
        uni3c = options.get(UNI3C_KEY)
        multitalk = options.get(MULTITALK_KEY)
        own_cache = options.get(ANIMATE2_CACHE_KEY)
        animate2_cache = options.pop("animate2_cache", None)
        if own_cache is not None:
            animate2_cache = own_cache
        for key in INTERNAL_KEYS:
            options.pop(key, None)
        # callbacks/wrappers в transformer_options исполняет ComfyUI на host (до/после этого вызова), в workers
        # они не передаются. Без allow_host_wrappers разрешены только известные host-only (WanAnimate2Cache cleanup,
        # InfiniteTalk), с ним — любые (как у H3).
        callbacks = strip_host_callbacks(options.pop("callbacks", None))
        host = self.session.config.allow_host_wrappers
        if has_effect(callbacks) and not host:
            raise ValueError("Сторонние callbacks в transformer_options: включите allow_host_wrappers в PowerShardConfigTuning")
        if "wrappers" in options:
            options["wrappers"] = strip_host_safe_wrappers(strip_wan_host_wrappers(options["wrappers"]))
            if host:
                options.pop("wrappers")
        replace = options.get("patches_replace") or {}
        if any(replace.values()):
            raise ValueError("PowerShard Wan: patches_replace — сторонний патч блоков не переносится в workers; "
                             "для кэша шагов используйте 'PowerShard: Block Cache' или EasyCache")
        options.pop("patches_replace", None)
        rope = options.pop("rope_options", None)
        validate_options(options)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.pop("denoise_mask", None)  # уже применён host process_timestep (Wan 2.2 TI2V)
        unknown = sorted(set(kwargs) - set(FORWARD_CONDITIONING))
        if unknown:
            raise ValueError(f"PowerShard Wan: conditioning {unknown} не поддерживается (context_latents/Bernini...)")
        experts = self.session.experts
        if slot not in experts:
            raise ValueError(f"Эксперт {slot!r} не загружен; доступны {sorted(experts)}")
        self.session.note_expert(slot)
        staged = self._stage_conditioning(context)
        for key, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.numel() * value.element_size() >= STAGE_BYTES:
                kwargs[key] = self._stage_conditioning(value, key + "_")
            elif isinstance(value, (list, tuple)) and not all(isinstance(v, (int, float)) for v in value):
                raise ValueError(f"PowerShard Wan: conditioning {key} неожиданного типа")
        extra = dict(kwargs, transformer_options=options, _powershard_expert=slot)
        if rope:
            extra["_powershard_rope_options"] = {k: float(v) for k, v in dict(rope).items()}
        if uni3c:
            spec = {k: v for k, v in uni3c.items() if k != "render"}
            extra["_powershard_uni3c"] = spec
            extra["uni3c_render"] = self._stage_conditioning(uni3c["render"], "uni3c_render_")
        if multitalk:
            extra["_powershard_multitalk"] = {k: v for k, v in multitalk.items() if k not in ("audio", "masks")}
            extra["multitalk_audio"] = self._stage_conditioning(multitalk["audio"].float(), "multitalk_audio_")
            if multitalk.get("masks") is not None:
                extra["multitalk_masks"] = multitalk["masks"].float().cpu()
        if animate2_cache is not None and kwargs.get("pose_latents") is not None and "context_window" not in full:
            # Как native: при context windows pose-кэш не используется (у каждого окна свой pose).
            extra["_powershard_animate2_cache"] = self.animate2_cache_spec(animate2_cache, dict(kwargs, _context=staged))
        cache = full.get("powershard_block_cache")
        if cache is not None:
            extra["_powershard_block_cache"] = cache.spec(full, timestep, [tuple(x.shape)], extra=(slot,))
        nag = full.get("powershard_nag")
        if nag is not None and nag.active(full, timestep):
            extra["_powershard_nag"] = nag.params()
            extra["nag_context"] = self._stage_conditioning(nag.context[:1].float(), "nag_context_")
        riflex = full.get("powershard_riflex")
        if riflex is not None:
            extra["_powershard_riflex"] = riflex.params()
        return self._call("forward", (x, timestep.float(), staged), extra, x.device)

    @staticmethod
    def animate2_cache_spec(cache, kwargs):
        """WanAnimate2Cache -> кэш входов pose branch в workers (ключ: содержимое pose[:1] + pose prompt)."""
        parts = []
        if isinstance(cache, dict):  # PowerShard Wan: Animate2 Cache
            ident, dtype = cache["id"], cache.get("dtype", "fp16")
        else:  # native WanAnimate2Cache (host PoseBranchCache не используется, только его идентичность/dtype)
            ident, dtype = f"{id(cache):x}", str(getattr(cache, "dtype", "default"))
        names = ["pose_latents", "context_pose" if kwargs.get("context_pose") is not None else "_context",
                 "clip_fea_pose" if kwargs.get("clip_fea_pose") is not None else "clip_fea"]
        for name in names:  # без pose-специфичных входов pose branch берёт текст/CLIP генерации
            value = kwargs.get(name)
            if isinstance(value, StagedTensor):
                parts.append(value.key)
            elif isinstance(value, torch.Tensor):
                parts.append(content_hash(value[:1].detach().to("cpu")))
        return dict(id=ident, key="|".join(parts), dtype=dtype)


_REMOTE_CLASSES = {}


def remote_class(base):
    if base not in _REMOTE_CLASSES:
        class RemoteWan(base):
            def get_dtype_inference(self):
                # Worker держит FP32 residual; host не сужает latent/context до FP16.
                return torch.float32

            def memory_required(self, input_shape, cond_shapes={}):
                # Activations Wan живут в VRAM workers, не в host cuda:0. Большая native
                # оценка заставила бы ComfyUI выгружать (парковать) сам PowerShard model
                # перед каждым sampler. Host держит лишь latents/conds в FP32.
                import math
                return 8 * 4 * math.prod(input_shape)

            def extra_conds(self, **kwargs):
                for key in ("control", "hooks", "gligen", "additional_models"):
                    if kwargs.get(key) is not None:
                        raise ValueError(f"PowerShard Wan не поддерживает conditioning {key}")
                out = super().extra_conds(**kwargs)
                if "context_latents" in out:
                    raise ValueError("PowerShard Wan: in-context latents (Bernini) пока не поддерживаются")
                return out
        RemoteWan.__name__ = RemoteWan.__qualname__ = "PowerShardRemote" + base.__name__
        _REMOTE_CLASSES[base] = RemoteWan
    return _REMOTE_CLASSES[base]


class WanPatcher(PowerShardPatcher):
    """Наследует wrappers (guard + run boundary) и memory accounting PowerShard."""

    def clone(self, *args, **kwargs):
        # Без clone_structure: clones делят одну модель-proxy -> ComfyUI is_clone().
        result = comfy.model_patcher.ModelPatcher.clone(self, *args, **kwargs)
        if hasattr(self, "powershard_memory_plan"):
            result.powershard_memory_plan = dict(self.powershard_memory_plan)
        return result

    def with_uni3c(self, spec):
        """Uni3C ControlNet: спецификация (путь, сила, sigma-окно, render latent) в transformer_options."""
        result = self.clone()
        result.model_options.setdefault("transformer_options", {})[UNI3C_KEY] = dict(spec)
        return result

    def with_multitalk(self, spec):
        """InfiniteTalk: путь model patch, audio_scale, audio_embeds (host audio_proj) и маски говорящих."""
        result = self.clone()
        result.model_options.setdefault("transformer_options", {})[MULTITALK_KEY] = dict(spec)
        return result

    def with_expert(self, slot, boundary=None):
        result = self.clone()
        options = result.model_options.setdefault("transformer_options", {})
        options[EXPERT_KEY] = slot
        if boundary is not None:
            options[BOUNDARY_KEY] = float(boundary)
        return result

    def with_h3_patch(self, patch):
        raise ValueError("H3 patch nodes не применяются к Wan; используйте PowerShard Wan Options в loader")

    def with_spectrum(self, config):
        raise ValueError("Spectrum реализован только для MiniMax H3")

    def add_patches(self, patches, *args, **kwargs):
        if patches:
            raise ValueError("Native LoRA loader не применим к FSDP Wan: подключите 'PowerShard Wan LoRA' ко входу lora "
                             "Wan loader — LoRA сливается в локальные shards при загрузке")
        return []

    def validate(self):
        if self.patches or self.hook_patches or self.weight_wrapper_patches or self.injections:
            raise ValueError("LoRA/hooks/weight patches через native nodes не поддерживаются; используйте PowerShard Wan LoRA")
        if self.forced_hooks is not None or self.additional_models:
            raise ValueError("Дополнительные модели/hooks не поддерживаются")
        from .accel import HOST_ONLY_TRANSFORMER_KEYS, check_easycache_holder, host_safe_model_options, strip_host_safe_wrappers
        if not self.session.config.allow_host_wrappers:
            for family in (strip_host_safe_wrappers(strip_wan_host_wrappers(strip_internal_wrappers(self.wrappers))),
                           self.host_callbacks()):
                if any(v for groups in family.values() for v in groups.values()):
                    raise ValueError("Сторонние callbacks/wrappers: установите allow_host_wrappers=True в PowerShard-конфиге "
                                     "(без этого допущены EasyCache, LazyCache, Context Windows)")
        if set(self.object_patches) - {"model_sampling"}:
            raise ValueError(f"Object patches {sorted(set(self.object_patches) - {'model_sampling'})} не переносятся в workers "
                             "(TeaCache/MagCache/SageAttention-патчи forward). Используйте 'PowerShard: Block Cache' и "
                             "attention backend в PowerShardConfig")
        if any(has_effect(v) for k, v in host_safe_model_options(self.model_options).items()
               if k not in {"transformer_options", "to_load_options"}):
            raise ValueError("Сторонние model_options не поддерживаются")
        if self.model_options.get("transformer_options", {}).get("easycache") is not None:
            check_easycache_holder(self.model_options["transformer_options"]["easycache"])
        to_load = dict(self.model_options.get("to_load_options", {}))
        if "wrappers" in to_load:
            to_load["wrappers"] = strip_internal_wrappers(to_load["wrappers"])
        if has_effect(to_load):
            raise ValueError("Сторонние to_load_options не поддерживаются")
        error = native_patch_error(self.model_options.get("transformer_options", {}).get("patches"))
        if error:
            raise ValueError(error)
        options = {k: v for k, v in self.model_options.get("transformer_options", {}).items()
                   if k not in INTERNAL_KEYS + ("rope_options", "animate2_cache", "callbacks") + HOST_ONLY_TRANSFORMER_KEYS}
        if "wrappers" in options:
            options["wrappers"] = strip_host_safe_wrappers(strip_wan_host_wrappers(options["wrappers"]))
            if self.session.config.allow_host_wrappers:
                options.pop("wrappers")
        validate_options(options)

    def host_callbacks(self):
        """Callbacks без host-only cleanup WanAnimate2Cache (освобождает лишь host PoseBranchCache)."""
        callbacks = {kind: dict(keyed) for kind, keyed in self.callbacks.items()}
        options = self.model_options.get("transformer_options", {})
        if "animate2_cache" in options:
            keyed = callbacks.get("on_cleanup", {})
            keyed[None] = [f for f in keyed.get(None, []) if getattr(f, "__qualname__", "").split(".")[0] != "WanAnimate2Cache"]
        return callbacks

    def cleanup(self):
        comfy.model_patcher.ModelPatcher.cleanup(self)
        session = self.session
        # high-noise проход MoE: не закрывать workers до low-noise прохода.
        if session.config.release_after_sampling and not session.draining and not session.should_defer():
            session.close()


def host_model(geometry, session):
    unet_config = comfy_unet_config(geometry)
    model_config = comfy.model_detection.model_config_from_unet_config(unet_config)
    if model_config is None:
        raise ValueError(f"ComfyUI не распознал Wan конфигурацию: {unet_config}")
    model_config.manual_cast_dtype = torch.float16
    native = model_config.get_model({}, device=torch.device("cpu"))
    native.__class__ = remote_class(type(native))
    native.diffusion_model = WanDiffusionProxy(session, geometry)
    native.eval().requires_grad_(False)
    return native


def load_wan(experts, config, options, loras=(), report_dir=None, boundary=None):
    """experts: {"main": path} или {"high": path, "low": path}. Возвращает базовый WanPatcher."""
    from .devices import resolve_gpu_selection
    from .patch_config import H3PatchConfig
    from .source_guard import require_signature
    from .wan_config import lora_plan_for
    from .wan_lora import plan_lora, safetensors_header
    from .wan_runtime import reusable_wan_session
    import comfy.ldm.wan.model as wan_model
    require_signature(wan_model.WanModel.rope_encode, ("t", "h", "w", "device", "dtype", "transformer_options"),
                      "WanModel.rope_encode")
    checkpoints = {slot: WanCheckpoint(path, options.model_type) for slot, path in experts.items()}
    geometries = {slot: c.model_config() for slot, c in checkpoints.items()}
    first_slot = next(iter(geometries))
    geometry = geometries[first_slot]
    for slot, g in geometries.items():
        if not same_geometry(geometry, g):
            raise ValueError(f"Эксперты {first_slot} и {slot} несовместимы (dim/heads/layers/in/out)")
    from dataclasses import replace
    from .quant_formats import resolve_weight_dtype
    for slot, c in checkpoints.items():
        if config.precision == "int8_fp16" and options.weight_format == "dequantize" and not c.quantization():
            raise ValueError(f"{slot}: precision=int8_fp16 (native int8 ядра) требует int8_tensorwise checkpoint, а "
                             f"{Path(c.path).name} — {c.storage()['kind']}. Выберите precision=fp16: любые форматы "
                             "(bf16/fp8/int8/nvfp4/mxfp8/int4) деквантуются по локальным строкам при загрузке")
    weight_dtype = resolve_weight_dtype(options.weight_dtype, resolve_gpu_selection(config.gpu_ids))
    if config.precision == "int8_fp16" and options.weight_format == "dequantize":
        weight_dtype = "fp16"
    options = replace(options, weight_dtype=weight_dtype, fp16_safe=options.fp16_safe and weight_dtype == "fp16")
    lora_plan = {}
    for slot, c in checkpoints.items():
        plan = lora_plan_for(slot, loras)
        if plan and config.precision == "int8_fp16" and options.weight_format == "dequantize":
            raise ValueError("LoRA нельзя слить в INT8 checkpoint")
        for item in plan:  # ранняя проверка ключей до запуска workers
            plan_lora(safetensors_header(item["path"]), c.tensors, label=f"{slot}: {Path(item['path']).name}")
        if plan:
            lora_plan[slot] = plan
    comfy_path = Path(comfy.model_base.__file__).resolve().parents[1]
    role_options = dict(experts={slot: str(c.path) for slot, c in checkpoints.items()},
                        loras=lora_plan, options=options.to_dict(),
                        families={slot: describe_family(g) for slot, g in geometries.items()})
    session = reusable_wan_session(str(checkpoints[first_slot].path), config, comfy_path, report_dir,
                                   patch=H3PatchConfig(enabled=True, fp16_safe=options.fp16_safe,
                                                       debug_finite=options.debug_finite,
                                                       mlp_chunk_tokens=options.mlp_chunk_tokens,
                                                       mlp_chunk_mode=options.mlp_chunk_mode),
                                   role_options=role_options)
    model = host_model(geometry, session)
    selected = resolve_gpu_selection(config.gpu_ids)
    plans = {slot: memory_plan(c, len(selected)) for slot, c in checkpoints.items()}
    resident = list(plans.values()) if options.moe_residency == "both" else [max(plans.values(), key=lambda p: p["shard_bytes_lower_bound"])]
    size = sum(p["shard_bytes_lower_bound"] for p in resident) if config.weight_placement == "gpu" else 0
    size += (1 + config.prefetch_blocks) * max(p["largest_group_bytes_upper_bound"] for p in plans.values())
    plan = dict(experts=plans, host_gpu_parameter_budget_bytes=size, residency=options.moe_residency,
                weight_placement=config.weight_placement, family=role_options["families"])
    patcher = WanPatcher(model, torch.device("cuda", int(selected[0]["user_id"])), torch.device("cpu"), size=size)
    patcher.powershard_memory_plan = plan
    if boundary is not None:
        patcher.powershard_wan_boundary = float(boundary)
    return patcher
