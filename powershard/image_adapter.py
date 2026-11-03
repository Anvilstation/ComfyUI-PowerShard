"""Host: родной BaseModel image-модели (SDXL, Flux, Z-Image, Qwen-Image, MOSS-Image, ...) без весов генератора.

BaseModel строится самим ComfyUI на meta-устройстве (model_sampling, latent_format, extra_conds — родные),
его diffusion_model заменяется proxy: forward уходит RPC в workers. Атрибуты модели, которые родной host-код
читает (img_in.weight.shape, flow_map, patch_size...), отдаёт meta-копия модели; её методы preprocess_*
(Flux.3, MOSS-Image refs) выполняются в workers.
"""
import functools
import uuid
import math
from pathlib import Path

import torch
import comfy.model_management
import comfy.model_patcher

from .comfy_adapter import PowerShardPatcher
from .conditioning_cache import content_hash
from .host_guard import strip_internal_wrappers
from .lifecycle import BoundaryPatcherMixin
from .wire import StagedTensor, has_effect, validate_options

STAGE_BYTES = 1 << 20
REMOTE_KEY = "__powershard_remote__"
PASS_OPTIONS = ("sample_sigmas", "sigmas", "cond_or_uncond", "uuids")
NOOP_METHODS = ("reset_prefix_cache",)
PARALLEL_MODES = ("auto", "sequence", "batch", "replicate")


def slg_skip_blocks(replace):
    """patches_replace["dit"] от Skip Layer Guidance (родные SkipLayerGuidanceDiT / ...Simple / SD3): функция
    возвращает вход блока без изменений, т.е. блок пропускается. В workers передаётся только список блоков, там
    ставится такой же пропуск. Другие замены блоков не переносятся."""
    out = []
    for kind, entries in (replace or {}).items():
        if not entries:
            continue
        for key, fn in entries.items():
            module = getattr(getattr(fn, "func", fn), "__module__", "") or ""
            if kind != "dit" or not module.endswith("nodes_slg") or not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f"PowerShard image: patches_replace {kind}/{key} ({module}) не переносится в workers "
                                 "(поддержан только Skip Layer Guidance)")
            out.append([str(key[0]), int(key[1])])
    return sorted(out)


class RemoteTensor:
    """Ссылка на большой тензор, оставшийся в workers (MOSS-Image ref K/V)."""

    def __init__(self, ident, shape, dtype):
        # без атрибута dtype: BaseModel._apply_model приводит к dtype всё, у чего он есть (cast_to на host)
        self.ident, self.shape, self._ps_dtype = ident, torch.Size(shape), dtype

    def wire(self):
        return {REMOTE_KEY: self.ident}

    def __repr__(self):
        return f"RemoteTensor({self.ident[:8]}, {list(self.shape)})"


def _to_wire(value):
    if isinstance(value, RemoteTensor):
        return value.wire()
    if isinstance(value, dict):
        return {k: _to_wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_wire(v) for v in value)
    return value


def _from_wire(value, device):
    if isinstance(value, dict):
        if REMOTE_KEY in value:
            return RemoteTensor(value[REMOTE_KEY], value.get("shape", []), value.get("dtype"))
        return {k: _from_wire(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_from_wire(v, device) for v in value)
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _first_device(values):
    for v in values:
        if isinstance(v, torch.Tensor):
            return v.device
        if isinstance(v, (list, tuple)):
            d = _first_device(v)
            if d is not None:
                return d
    return None


class ImageProxy(torch.nn.Module):
    def __init__(self, session, meta_network, dtype):
        super().__init__()
        self.session = session
        self.dtype = dtype
        self.__dict__["_meta"] = meta_network  # не подмодуль: ComfyUI не должен переносить meta-тензоры

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        meta = self.__dict__.get("_meta")
        if meta is None or name.startswith("__"):
            raise AttributeError(name)
        value = getattr(meta, name)
        if callable(value) and not isinstance(value, torch.nn.Module) and hasattr(type(meta), name):
            if name in NOOP_METHODS:
                return lambda *a, **k: None
            if name.startswith("preprocess"):
                return functools.partial(self._remote_method, name)
            return functools.partial(self._unsupported, name)
        return value

    @staticmethod
    def _unsupported(name, *args, **kwargs):
        raise ValueError(f"PowerShard image: метод модели {name}() вызывается на host и не поддерживается")

    def _call(self, command, args, kwargs):
        return self.session.call(command, args, kwargs,
                                 cancel=comfy.model_management.throw_exception_if_processing_interrupted)

    def _remote_method(self, name, *args, **kwargs):
        device = _first_device(list(args) + list(kwargs.values())) or torch.device("cpu")
        # id ссылок на тензоры, оставшиеся в workers, задаёт host: одинаковый на всех rank'ах
        result = self._call("method", _to_wire(tuple(args)), dict(_to_wire(kwargs), _powershard_method=name,
                                                                  _powershard_remote_id=uuid.uuid4().hex))
        return _from_wire(result, device)

    def _stage(self, value, prefix):
        cpu = value.detach().to("cpu")
        key = prefix + content_hash(cpu)
        self.session.stage_tensors({key: cpu})
        return StagedTensor(key, index=0)

    def forward(self, x, timestep, context=None, control=None, transformer_options=None, **kwargs):
        from .accel import run_diffusion_wrappers
        options = dict(transformer_options or {})
        return run_diffusion_wrappers(self._remote, self, (x, timestep, context),
                                      dict(control=control, transformer_options=options, **kwargs), options)

    def _remote(self, x, timestep, context=None, control=None, transformer_options=None, **kwargs):
        from .accel import HOST_ONLY_TRANSFORMER_KEYS
        full = dict(transformer_options or {})
        for key in ("powershard_block_cache", "powershard_nag", "powershard_riflex"):
            if full.get(key) is not None:
                raise ValueError("PowerShard Block Cache / NAG / RIFLEx реализованы для Wan/LTX; для image-моделей "
                                 "используйте EasyCache / родные CFG-ноды")
        skip_blocks = slg_skip_blocks(full.get("patches_replace"))
        if has_effect(full.get("patches")):
            raise ValueError("PowerShard image: model patches (IP-Adapter/PuLID/attention patches) не переносятся в "
                             "workers; используйте их с родным loader")
        if has_effect(full.get("callbacks")) and not self.session.config.allow_host_wrappers:
            raise ValueError("Сторонние callbacks в transformer_options: включите allow_host_wrappers в "
                             "PowerShardConfigTuning")
        for key in list(full):
            if key in PASS_OPTIONS or key in HOST_ONLY_TRANSFORMER_KEYS or key in ("wrappers", "callbacks", "patches",
                                                                                  "patches_replace", "rope_options"):
                continue
            if has_effect(full[key]) and not isinstance(full[key], (int, float, str, bool)):
                raise ValueError(f"PowerShard image: transformer_options[{key!r}] не переносится в workers")
        options = {k: full[k] for k in PASS_OPTIONS if k in full}
        validate_options(options)
        extra = {k: v for k, v in kwargs.items() if v is not None}
        if control is not None:
            extra["control"] = control
        if skip_blocks:
            extra["_powershard_skip_blocks"] = skip_blocks
        rope = full.get("rope_options")
        if rope:
            extra["_powershard_rope_options"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                                                 for k, v in dict(rope).items()}
        for key, value in list(extra.items()):
            if isinstance(value, torch.Tensor) and value.numel() * value.element_size() >= STAGE_BYTES:
                extra[key] = self._stage(value, key + "_")
        staged = context
        if isinstance(context, torch.Tensor) and context.numel() * context.element_size() >= STAGE_BYTES:
            staged = self._stage(context, "context_")
        extra = _to_wire(extra)
        extra["transformer_options"] = options
        try:
            result = self._call("forward", (x, timestep.float(), staged), extra)
        except TypeError as error:
            raise ValueError(f"PowerShard image: conditioning не сериализуется ({error})") from error
        return result.to(x.device)


def image_guard_sampling(executor, model, noise_shape, conds, **kwargs):
    """Как host_guard.guard_sampling, но ControlNet разрешён: он считается на host, workers получают его выход."""
    model.validate()
    for branch in conds.values():
        for cond in branch:
            for key in ("hooks", "gligen", "additional_models"):
                if cond.get(key) is not None:
                    raise ValueError(f"PowerShard image: неподдерживаемый {key} в conditioning; workers не запущены")
    return executor(model, noise_shape, conds, **kwargs)


def _strip_own(wrappers):
    out = {}
    for kind, keyed in (wrappers or {}).items():
        out[kind] = {}
        for key, functions in keyed.items():
            kept = [f for f in functions if not (kind == "prepare_sampling" and key == "powershard"
                                                 and f is image_guard_sampling)]
            if kept:
                out[kind][key] = kept
    return out


_REMOTE_CLASSES = {}


def remote_class(base):
    if base not in _REMOTE_CLASSES:
        class RemoteImage(base):
            def memory_required(self, input_shape, cond_shapes={}):
                # Активации живут в VRAM workers; host держит лишь latents/conds.
                return 8 * 4 * math.prod(input_shape)

            def get_writable_model(self, *args, **kwargs):
                return None  # KV/prefix-кэши родного host-кода PowerShard не использует

            def extra_conds(self, **kwargs):
                for key in ("hooks", "gligen", "additional_models"):
                    if kwargs.get(key) is not None:
                        raise ValueError(f"PowerShard image не поддерживает conditioning {key}")
                return super().extra_conds(**kwargs)
        RemoteImage.__name__ = RemoteImage.__qualname__ = "PowerShardRemote" + base.__name__
        _REMOTE_CLASSES[base] = RemoteImage
    return _REMOTE_CLASSES[base]


class ImagePatcher(BoundaryPatcherMixin, PowerShardPatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrappers.setdefault("prepare_sampling", {})["powershard"] = [image_guard_sampling]
        self._install_boundary()

    def clone(self, *args, **kwargs):
        result = comfy.model_patcher.ModelPatcher.clone(self, *args, **kwargs)
        if hasattr(self, "powershard_memory_plan"):
            result.powershard_memory_plan = dict(self.powershard_memory_plan)
        return result

    def with_h3_patch(self, patch):
        raise ValueError("H3 patch nodes не применяются к image-моделям")

    def with_spectrum(self, config):
        raise ValueError("Spectrum реализован только для MiniMax H3")

    def add_patches(self, patches, *args, **kwargs):
        if patches:
            raise ValueError("Родной LoRA loader не применим к FSDP-модели PowerShard: подключите «PowerShard: LoRA» "
                             "ко входу lora loader-а — LoRA сливается в shards при загрузке")
        return []

    def validate(self):
        if self.patches or self.hook_patches or self.weight_wrapper_patches or self.injections:
            raise ValueError("LoRA/hooks/weight patches через родные ноды не поддерживаются; используйте «PowerShard: LoRA»")
        if self.forced_hooks is not None or self.additional_models:
            raise ValueError("Дополнительные модели/hooks не поддерживаются")
        from .accel import HOST_ONLY_TRANSFORMER_KEYS, check_easycache_holder, host_safe_model_options, strip_host_safe_wrappers
        if not self.session.config.allow_host_wrappers:
            for family in (strip_host_safe_wrappers(strip_internal_wrappers(_strip_own(self.wrappers))), self.callbacks):
                if any(v for groups in family.values() for v in groups.values()):
                    raise ValueError("Сторонние callbacks/wrappers: установите allow_host_wrappers=True в "
                                     "PowerShardConfigTuning (без этого допущены EasyCache, LazyCache, Context Windows)")
        if set(self.object_patches) - {"model_sampling"}:
            raise ValueError(f"Object patches {sorted(set(self.object_patches) - {'model_sampling'})} не переносятся "
                             "в workers (патчи forward/attention сторонних нод)")
        if any(has_effect(v) for k, v in host_safe_model_options(self.model_options).items()
               if k not in {"transformer_options", "to_load_options"}):
            raise ValueError("Сторонние model_options не поддерживаются")
        options = self.model_options.get("transformer_options", {})
        if options.get("easycache") is not None:
            check_easycache_holder(options["easycache"])
        if any((options.get("patches_replace") or {}).values()) or has_effect(options.get("patches")):
            raise ValueError("Model patches / patches_replace (IP-Adapter, PuLID, attention patches) не переносятся "
                             "в workers PowerShard")
        to_load = dict(self.model_options.get("to_load_options", {}))
        if "wrappers" in to_load:
            to_load["wrappers"] = strip_internal_wrappers(to_load["wrappers"])
        if has_effect(to_load):
            raise ValueError("Сторонние to_load_options не поддерживаются")
        rest = {k: v for k, v in options.items()
                if k not in HOST_ONLY_TRANSFORMER_KEYS + ("rope_options", "callbacks", "patches", "patches_replace")}
        if "wrappers" in rest:
            rest["wrappers"] = strip_host_safe_wrappers(rest["wrappers"])
            if self.session.config.allow_host_wrappers:
                rest.pop("wrappers")
        validate_options({k: v for k, v in rest.items() if k in PASS_OPTIONS or k == "wrappers" or has_effect(v)})

    def cleanup(self):
        comfy.model_patcher.ModelPatcher.cleanup(self)
        session = self.session
        if session.config.release_after_sampling and not session.draining:
            session.close()


# ------------------------------------------------------------------ загрузка
def decide_dtypes(supported, weight_dtype, capabilities):
    """(хранение, вычисление) для всех GPU набора: bf16 только при sm80+ на каждой карте."""
    names = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
    allowed = [names[d] for d in supported if d in names] or ["fp16", "bf16", "fp32"]
    bf16_ok = bool(capabilities) and min(tuple(c) for c in capabilities) >= (8, 0)
    if weight_dtype in ("fp16", "bf16", "fp32"):
        storage = weight_dtype
        compute = "fp32" if storage == "bf16" and not bf16_ok else storage
        return storage, compute
    if bf16_ok and "bf16" in allowed:
        return "bf16", "bf16"
    if "fp16" in allowed:
        return "fp16", "fp16"
    if "bf16" in allowed:  # V100 и bf16-модели (Z-Image, Qwen-Image, MOSS): память как bf16, точное FP32 вычисление
        return "bf16", "fp32"
    return "fp32", "fp32"


def load_image(path, config, parallel="auto", weights=None, loras=(), report_dir=None):
    from .common_nodes import weights_or_default
    from .devices import resolve_gpu_selection
    from .image_backend import DTYPES, build_native
    from .patch_config import H3PatchConfig
    from .quant_formats import validate_quant_choice
    from .wan_runtime import reusable_wan_session
    if parallel not in PARALLEL_MODES:
        raise ValueError(f"parallel: {' / '.join(PARALLEL_MODES)}")
    w = weights_or_default(weights)
    validate_quant_choice(w.weight_format, w.compute)
    if config.precision == "int8_fp16":
        raise ValueError("precision=int8_fp16 (Int8Linear H3/Wan) не применяется к image-моделям: выберите fp16 и "
                         "формат int8 в узле «PowerShard: формат весов / вычисление»")
    selected = resolve_gpu_selection(config.gpu_ids)
    caps = [tuple(d.get("capability") or (7, 0)) for d in selected]
    model, model_config, sd, _, header, prefix = build_native(path, torch.float16, torch.float16)
    storage, compute = decide_dtypes(model_config.supported_inference_dtypes, w.weight_dtype, caps)
    model_config.set_inference_dtype(DTYPES[storage], None if compute == storage else DTYPES[compute])
    parameters = sum(p.numel() for p in model.diffusion_model.parameters())
    meta_network = model.diffusion_model
    model.__class__ = remote_class(type(model))
    model.diffusion_model = ImageProxy(None, meta_network, DTYPES[compute])
    model.eval().requires_grad_(False)
    loras = [dict(path=str(l.path), strength=float(l.strength)) for l in loras or ()]
    comfy_path = Path(comfy.model_management.__file__).resolve().parents[1]
    role_options = dict(kind="image", experts={"main": str(Path(path).resolve())}, loras={"main": loras},
                        options=dict(storage=storage, compute=compute, parallel=parallel, weight_format=w.weight_format,
                                     compute_mode=w.compute, fp16_accumulation=bool(w.fp16_accumulation)),
                        families={"main": type(model).__name__.removeprefix("PowerShardRemote")})
    session = reusable_wan_session(str(Path(path).resolve()), config, comfy_path, report_dir,
                                   patch=H3PatchConfig(enabled=True), role_options=role_options)
    model.diffusion_model.session = session
    bytes_per = {"fp16": 2, "bf16": 2, "fp32": 4}[storage]
    shard = math.ceil(parameters * bytes_per / max(1, len(selected)))
    size = shard if config.weight_placement == "gpu" else 0
    plan = dict(parameters=parameters, storage=storage, compute=compute, parallel=parallel,
                shard_bytes_estimate=shard, family=role_options["families"]["main"], key_prefix=prefix,
                weight_placement=config.weight_placement)
    patcher = ImagePatcher(model, torch.device("cuda", int(selected[0]["user_id"])), torch.device("cpu"), size=size)
    patcher.powershard_memory_plan = plan
    del sd, header
    return patcher
