"""Родной MiniMaxH3 BaseModel/ModelPatcher; удалён только владелец diffusion weights."""
import copy
from pathlib import Path
import torch
import comfy.model_base
import comfy.model_patcher
import comfy.model_management
import comfy.supported_models
from .checkpoint import Checkpoint, memory_plan
from .runtime import reusable_session
from .wire import validate_options, has_effect, StagedTensor
from .conditioning_cache import content_hash
from .source_guard import verify_comfy
from .host_guard import guard_sampling, strip_internal_wrappers


class DiffusionProxy(torch.nn.Module):
    def __init__(self, session, model_config):
        super().__init__()
        self.session = session
        self.dtype = torch.float16
        self.hidden_size = model_config["hidden_size"]
        self.patch_size = (1,2,2)
        self.latents_dim, self.audio_latents_dim = model_config["latents_dim"], model_config["audio_latents_dim"]
        self.sigma_shift_video, self.sigma_shift_audio = 12., 3.
        self.shape_config = dict(model_config)

    def _call(self, command, args, kwargs, device):
        def cancel():
            comfy.model_management.throw_exception_if_processing_interrupted()
        result = self.session.call(command, args, kwargs, cancel=cancel)
        def move(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return [move(y) for y in x]
        return move(result)

    def preprocess_text_embeds(self, text):
        return self._call("preprocess_text", (text,), {}, text.device)

    def _stage_conditioning(self, context):
        """Один раз за run пишет context в run-stage; дальше — StagedTensor.

        Тензоры на CPU (host-сторона модели); worker кэширует их на device.
        Stage очищается на end_run/close; каждый новый forward заново
        регистрирует conditioning, поэтому stale-ссылки невозможны. Ключ —
        content-hash (shape+bytes), а не id(): ComfyUI может переиспользовать
        аллокацию с тем же адресом под другой conditioning.
        """
        key = "context_" + content_hash(context.detach().to("cpu"))
        self.session.stage_tensors({key: context.detach().to("cpu")})
        return StagedTensor(key, index=0)

    def forward(self, x, timestep, context, control=None, transformer_options=None, **kwargs):
        if control is not None:
            raise ValueError("ControlNet пока не поддерживается PowerShard")
        validate_options(transformer_options or {})
        if self.session.role_options.get("spectrum",{}).get("enabled",False):
            from .spectrum_host import forward_metadata
            kwargs=dict(kwargs,_powershard_spectrum=forward_metadata(self.session,x,timestep,context,kwargs,transformer_options or {}))
        staged = self._stage_conditioning(context)
        return self._call("forward", (x, timestep, staged),
                          dict(transformer_options=transformer_options or {}, **kwargs), x[0].device)


class RemoteH3(comfy.model_base.MiniMaxH3):
    def get_dtype_inference(self):
        patch = getattr(self.diffusion_model.session, "patch", None)
        # Защита ДО cast в native extra_conds/_apply_model, не после появления Inf.
        return torch.float32 if patch is not None and patch.active else super().get_dtype_inference()

    def extra_conds(self, **kwargs):
        for key in ("control", "hooks", "gligen", "additional_models"):
            if kwargs.get(key) is not None:
                raise ValueError(f"H3 PowerShard не поддерживает conditioning {key}")
        return super().extra_conds(**kwargs)


class PowerShardPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_wrapper_with_key("prepare_sampling", "powershard", guard_sampling)
        from .spectrum_host import spectrum_outer_sample
        self.add_wrapper_with_key("outer_sample","powershard_run",spectrum_outer_sample)

    @property
    def session(self):
        return self.model.diffusion_model.session

    def clone(self, *args, **kwargs):
        result = super().clone(*args, **kwargs)
        # Proxy не владеет generator weights. Клонируется лишь маленькое дерево
        # ModelSampling/proxy и контейнеры; неизменная session может разделяться.
        result.model = clone_structure(result.model)
        if hasattr(self, "powershard_memory_plan"):
            result.powershard_memory_plan = dict(self.powershard_memory_plan)
        return result

    def with_h3_patch(self, patch):
        self.validate()
        result = self.clone()
        result.model.diffusion_model.session = self.session.with_patch(patch)
        result.model.diffusion_model.dtype = torch.float32 if patch.active else torch.float16
        return result

    def with_spectrum(self, config):
        self.validate()
        result=self.clone()
        result.model.diffusion_model.session=self.session.with_spectrum(config)
        # Общий sampler boundary уже установлен в __init__; конфигурация clone
        # определяет, отправлять ли Spectrum metadata в worker.
        return result

    def validate(self):
        if self.patches or self.hook_patches or self.weight_wrapper_patches or self.injections:
            raise ValueError("LoRA, hooks и weight patches ещё не поддерживаются")
        if self.forced_hooks is not None or self.additional_models:
            raise ValueError("Дополнительные модели/hooks не поддерживаются")
        # Сторонние wrappers/callbacks: по умолчанию отказ — они могут менять
        # тензоры между RPC (LoRA-стиль), и workers увидят не то, что host.
        # allow_host_wrappers=True (opt-in): выполняем их на host; наблюдатели
        # (логирование, превью, telemetry) работают, модификаторы — на риск.
        if not self.session.config.allow_host_wrappers:
            for family in (strip_internal_wrappers(self.wrappers), self.callbacks):
                if any(v for groups in family.values() for v in groups.values()):
                    raise ValueError("Сторонние callbacks/wrappers: установите allow_host_wrappers=True в PowerShard-конфиге "
                                     "(они выполнятся на host; wrappers, меняющие тензоры, могут дать неверный результат)")
        if set(self.object_patches) - {"model_sampling"}:
            raise ValueError("Разрешён только native model_sampling patch")
        if any(has_effect(v) for k,v in self.model_options.items() if k not in {"transformer_options", "to_load_options"}):
            raise ValueError("Сторонние model_options не поддерживаются")
        to_load = dict(self.model_options.get("to_load_options", {}))
        if "wrappers" in to_load:
            to_load["wrappers"] = strip_internal_wrappers(to_load["wrappers"])
        if has_effect(to_load):
            raise ValueError("Сторонние to_load_options не поддерживаются")
        validate_options(self.model_options.get("transformer_options", {}))

    def add_patches(self, patches, *args, **kwargs):
        if patches:
            raise ValueError("LoRA для FSDP H3 пока не реализована")
        return []

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        self.validate()
        # Реальная загрузка отложена до первого RPC после проверки conditioning.
        self.model.device = device_to or self.load_device
        self.model.current_patcher = self
        self.model.model_loaded_weight_memory = self.model_size()

    def loaded_size(self):
        if not self.session.running:
            return 0
        if self.session.idle_on_cpu:
            # Zero is a measured parking result, not a missing first-RPC sample.
            return self.session.last_memory
        # Worker allocations видны NVML/free VRAM, но не host torch allocator.
        # До первого worker-замера отдаём бюджет из memory plan (шарды +
        # active/prefetch groups), НЕ ноль: ComfyUI-менеджер с нулевой оценкой
        # перестаёт резервировать VRAM и забивает пул другими моделями.
        # После первого RPC — фактический allocated workers.
        plan = getattr(self, "powershard_memory_plan", None)
        if self.session.config.weight_placement == "ats":
            return (plan or {}).get("host_gpu_parameter_budget_bytes", 0)
        return self.session.last_memory or (plan or {}).get("host_gpu_parameter_budget_bytes", 0)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        previous = self.loaded_size()
        self.patch_model(device_to=device_to)
        return max(0, self.loaded_size()-previous)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        before = self.loaded_size()
        self.session.deactivate()
        self.model.model_loaded_weight_memory = 0
        return before

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        # ModelPatcher переносит только маленький native model_sampling и proxy без weights.
        self.session.deactivate()
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def cleanup(self):
        try:
            super().cleanup()
        finally:
            if self.session.config.release_after_sampling and not self.session.draining:
                self.session.close()  # Освободить выбранные GPU перед audio/video VAE.


def clone_structure(module, memo=None):
    memo = {} if memo is None else memo
    if id(module) in memo:
        return memo[id(module)]
    result = copy.copy(module)
    memo[id(module)] = result
    result._modules = {k: None if v is None else clone_structure(v, memo) for k, v in module._modules.items()}
    result._parameters = dict(module._parameters)
    result._buffers = dict(module._buffers)
    # Hook dictionaries не должны изменять источник при дальнейшем patching.
    for name, value in module.__dict__.items():
        if "hook" in name and isinstance(value, dict):
            setattr(result, name, value.copy())
    return result


def load_model(path, config, report_dir=None):
    comfy_path = Path(comfy.model_base.__file__).resolve().parents[1]
    verify_comfy(comfy_path)
    checkpoint = Checkpoint(path)
    shape = checkpoint.model_config()
    quant = checkpoint.quantization()
    if bool(quant) != (config.precision == "int8_fp16"):
        raise ValueError("Выбранный checkpoint не соответствует precision")
    model_config = comfy.supported_models.MiniMaxH3(dict(shape, image_model="minimax_h3", disable_unet_model_creation=True, dtype=torch.float16))
    model_config.manual_cast_dtype = torch.float16
    model = RemoteH3(model_config, device=torch.device("cpu"))
    from .patch_config import H3PatchConfig
    session = reusable_session(str(checkpoint.path), config, comfy_path, report_dir=report_dir,
                      patch=H3PatchConfig(enabled=True, mlp_chunk_mode="off", mlp_chunk_tokens=4096))
    model.diffusion_model = DiffusionProxy(session, shape)
    model.diffusion_model.dtype = torch.float32
    model.eval().requires_grad_(False)
    from .devices import resolve_gpu_selection
    selected = resolve_gpu_selection(config.gpu_ids)
    plan = memory_plan(checkpoint.tensors, bool(quant), len(selected))
    # CPUOffloadPolicy shards не занимают постоянную GPU память. В budget
    # остаются собранные current/prefetch groups. Это estimate, не measured VRAM.
    size = (plan["shard_bytes_lower_bound"] if config.weight_placement == "gpu" else 0)
    size += (1+config.prefetch_blocks)*plan["largest_group_bytes_upper_bound"]
    plan["host_gpu_parameter_budget_bytes"] = size
    patcher = PowerShardPatcher(model, torch.device("cuda", int(selected[0]["user_id"])), torch.device("cpu"), size=size)
    patcher.powershard_memory_plan = plan
    return patcher
