"""Lazy imports позволяют проверить регистрацию нод без torch/CUDA/ComfyUI."""
import json
from pathlib import Path
from .config import DistributedConfig


class PowerShardConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gpu_ids": ("STRING", {"default": "all", "tooltip": "all или CUDA-visible индексы: 5,2,0. Порядок задаёт ranks."}),
            "weight_placement": (["gpu", "cpu", "ats"], {"default": "gpu", "tooltip": "gpu: постоянные shards в VRAM. cpu: shards в pinned RAM, активный блок на GPU. ats: CUDA Unified Memory shards, аппаратный ATS обязателен; экспериментальный."}),
            "precision": (["fp16", "int8_fp16"], {"default": "fp16"}),
            "attention_backend": (["auto", "vllm_flash_attn", "flash_attn", "sdpa", "math", "sageattention"], {"default": "auto", "tooltip": "auto проверяет custom vLLM FA, FA, SDPA, math на каждой карте. При разрешённом fallback отсутствующий flash_attn также проверяет vllm_flash_attn. auto не выбирает по скорости. Фактические вызовы по группам — в статусе/логах."}),
            "sequence_mode": (["token", "ulysses"], {"default": "token", "tooltip": "token: полный K/V gather. Ulysses: head all-to-all с padding, любое число GPU (3/4/5/6 и т.д.)."})}}
    RETURN_TYPES = ("POWERSHARD_CONFIG",)
    FUNCTION = "create"
    CATEGORY = "PowerShard"

    def create(self, gpu_ids="all", weight_placement="gpu", precision="fp16", attention_backend="auto", sequence_mode="token", **legacy):
        # Old API exports use named fields. Old UI JSON is migrated by the
        # frontend/CLI before positional widget deserialization.
        if legacy.get("cpu_offload") and weight_placement == "gpu":
            weight_placement = "cpu"
        allowed = set(DistributedConfig.__dataclass_fields__) - {"gpu_ids", "weight_placement", "precision", "attention_backend", "sequence_mode", "cpu_offload"}
        values = {k: v for k, v in legacy.items() if k in allowed}
        values.update(backend="fsdp2_sequence", memory_policy="auto")
        return (DistributedConfig(gpu_ids=gpu_ids, weight_placement=weight_placement,
                                  precision=precision, attention_backend=attention_backend,
                                  sequence_mode=sequence_mode, **values),)


class PowerShardConfigTuning:
    """Optional controls with real effects, kept out of the primary node."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": ("POWERSHARD_CONFIG",),
            "reserve_gib": ("FLOAT", {"default": 2., "min": 0., "max": 32.}),
            "prefetch_blocks": ([0, 1, 2], {"default": 0}),
            "numa_policy": (["none", "auto", "bind"], {"default": "auto"}),
            "strict_attention": ("BOOLEAN", {"default": False}),
            "allow_host_wrappers": ("BOOLEAN", {"default": False}),
            "pin_memory": ("BOOLEAN", {"default": True, "tooltip": "Pinned CPU shards для cpu. При дефиците RAM можно отключить; H2D станет медленнее."})},
            "optional": {
                "sequence_comm_dtype": (["fp32", "fp16"], {"default": "fp32", "tooltip": "FP32: путь старого SDPA baseline. FP16: меньше обмен, но дополнительный MAX all-reduce в каждом H3 attention block."}),
                "prefetch_policy": (["auto", "manual"], {"default": "auto", "tooltip": "Auto может уменьшить prefetch до 0; причина видна в логе. Manual сохраняет выбранные 1/2 блока, но может вызвать OOM."})}}
    RETURN_TYPES = ("POWERSHARD_CONFIG",)
    FUNCTION = "tune"
    CATEGORY = "PowerShard/Advanced"

    def tune(self, config, reserve_gib=2., prefetch_blocks=0, numa_policy="auto", keep_workers=None,
             strict_attention=False, allow_host_wrappers=False, pin_memory=True,
             sequence_comm_dtype=None, prefetch_policy=None):
        from dataclasses import replace
        return (replace(config, reserve_gib=reserve_gib, prefetch_blocks=int(prefetch_blocks),
                        numa_policy=numa_policy, release_after_sampling=config.release_after_sampling if keep_workers is None else not keep_workers,
                        allow_fallback=not strict_attention, allow_host_wrappers=allow_host_wrappers,
                        pin_memory=pin_memory,
                        sequence_comm_dtype=sequence_comm_dtype or config.sequence_comm_dtype,
                        memory_policy=prefetch_policy or config.memory_policy),)


class PowerShardH3Loader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"checkpoint": (folder_paths.get_filename_list("diffusion_models"),), "config": ("POWERSHARD_CONFIG",)},
                "optional": {"keep_in_memory": ("BOOLEAN", {"default": True,
                    "tooltip": "Между задачами сохранить локальные веса в RAM, освободить VRAM для VAE. GPU/ATS восстанавливаются из RAM без перечитывания весов checkpoint. Отмена текущего RPC завершает его в фоне; ошибка CUDA/NCCL требует перезагрузки."})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "PowerShard"
    def load(self, checkpoint, config, keep_in_memory=True):
        from dataclasses import replace
        import folder_paths
        from .comfy_adapter import load_model
        path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint)
        config=replace(config,release_after_sampling=not keep_in_memory)
        return (load_model(path, config, Path(folder_paths.get_output_directory())/"powershard"),)
    @classmethod
    def IS_CHANGED(cls, checkpoint, config, keep_in_memory=True):
        import folder_paths
        p = Path(folder_paths.get_full_path_or_raise("diffusion_models", checkpoint))
        st = p.stat()
        from .web_api import provider_stamp
        return (st.st_size, st.st_mtime_ns, repr(config), keep_in_memory, provider_stamp())


class PowerShardH3TextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"checkpoint": (folder_paths.get_filename_list("text_encoders"),),
                             "placement": (["cpu_fp32", "native_offload"],)}}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "PowerShard"
    def load(self, checkpoint, placement):
        import torch
        import comfy.sd
        import folder_paths
        if "qwen3vl_32b_minimax_h3" not in checkpoint:
            raise ValueError("Нужен родной qwen3vl_32b_minimax_h3 checkpoint")
        if placement == "cpu_fp32" and not checkpoint.endswith("_bf16.safetensors"):
            raise ValueError("CPU FP32 режим использует только BF16 source; INT8 text encoder требует отдельной проверки kernels")
        opts = {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu"), "dtype": torch.float32} if placement == "cpu_fp32" else {}
        clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", checkpoint)],
                                  embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                  clip_type=comfy.sd.CLIPType.MINIMAX, model_options=opts)
        return (clip,)


class PowerShardRelease:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)},"optional":{
            "preserve_qwen_cpu_shards":("BOOLEAN",{"default":True}),
            "clear_conditioning_cache":("BOOLEAN",{"default":False}),
            "preserve_h3_cpu_shards":("BOOLEAN",{"default":True})}}
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "report")
    FUNCTION = "release"
    CATEGORY = "PowerShard"
    def release(self, samples, preserve_qwen_cpu_shards=True, clear_conditioning_cache=False, preserve_h3_cpu_shards=True):
        from .runtime import release_all
        release_all(preserve_idle_cpu=preserve_qwen_cpu_shards,preserve_h3_cpu=preserve_h3_cpu_shards)
        if clear_conditioning_cache:
            from .conditioning_cache import clear_all_caches
            clear_all_caches()
        return (samples, "GPU-фаза освобождена. CPU shards сохранены при включённом удержании: H3="+str(preserve_h3_cpu_shards)+", Qwen="+str(preserve_qwen_cpu_shards))


class PowerShardH3QwenLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required":{"checkpoint":(folder_paths.get_filename_list("text_encoders"),),
            "config":("POWERSHARD_CONFIG",),"precision":(["fp16","int8_fp16"],),
            "idle_policy":(["release","cpu_shards","keep"],),
            "cache_mib":("INT",{"default":256,"min":0,"max":2147483647})}}
    RETURN_TYPES=("CLIP",)
    FUNCTION="load"
    CATEGORY="PowerShard"
    def load(self,checkpoint,config,precision="fp16",idle_policy="release",cache_mib=256,mlp_chunk_mode="off",mlp_chunk_tokens=4096):
        from dataclasses import replace
        import folder_paths
        from .qwen import QwenConfig
        from .qwen_adapter import load_qwen
        return (load_qwen(folder_paths.get_full_path_or_raise("text_encoders",checkpoint),replace(config,precision=precision),
            QwenConfig(idle_policy,cache_mib,mlp_chunk_mode,mlp_chunk_tokens),folder_paths.get_folder_paths("embeddings"),
            Path(folder_paths.get_output_directory())/"powershard"),)
    @classmethod
    def IS_CHANGED(cls,checkpoint,config,**kwargs):
        import folder_paths
        from .web_api import provider_stamp
        path=Path(folder_paths.get_full_path_or_raise("text_encoders",checkpoint));st=path.stat()
        return st.st_size,st.st_mtime_ns,repr(config),repr(kwargs),provider_stamp()


class PowerShardH3FP16Patcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "fp16_safe": ("BOOLEAN", {"default": True}),
                             "debug_finite": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "PowerShard/Patchers"

    def patch(self, model, fp16_safe=True, debug_finite=False, **legacy):
        from dataclasses import replace
        from .comfy_adapter import PowerShardPatcher
        if not isinstance(model, PowerShardPatcher):
            raise ValueError("Нужен MODEL из PowerShard H3 Loader")
        previous = model.session.patch
        policy = replace(previous, enabled=legacy.get("enabled", True), fp16_safe=fp16_safe,
                         debug_finite=debug_finite,
                         mlp_chunk_tokens=legacy.get("mlp_chunk_tokens", previous.mlp_chunk_tokens),
                         mlp_chunk_mode=legacy.get("mlp_chunk_mode", previous.mlp_chunk_mode))
        return (model.with_h3_patch(policy),)


class PowerShardH3MLP:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "mode": (["off", "auto", "manual"], {"default":"off"}),
            "chunk_tokens": ("INT", {"default": 4096, "min": 1, "max": 2147483647,
                "tooltip": "manual: точный лимит; auto: расчёт по VRAM; off: полный MLP. FP16 Safe остаётся независимым."})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "PowerShard/MLP"

    def patch(self, model, mode="off", chunk_tokens=4096):
        from dataclasses import replace
        from .comfy_adapter import PowerShardPatcher
        if not isinstance(model, PowerShardPatcher):
            raise ValueError("H3 MLP требует MODEL PowerShard")
        return (model.with_h3_patch(replace(model.session.patch, mlp_chunk_mode=mode,
                                           mlp_chunk_tokens=chunk_tokens)),)


class PowerShardQwenMLP:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "mode": (["off", "auto", "manual"], {"default":"off"}),
            "chunk_tokens": ("INT", {"default": 4096, "min": 1, "max": 2147483647})}}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"
    CATEGORY = "PowerShard/MLP"

    def patch(self, clip, mode="off", chunk_tokens=4096):
        from .qwen_adapter import DistributedQwenCLIP
        if not isinstance(clip, DistributedQwenCLIP):
            raise ValueError("Qwen MLP требует CLIP из PowerShard H3 Qwen Loader")
        return (clip.with_mlp(mode, chunk_tokens),)


class PowerShardDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"refresh": ("INT", {"default": 0})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "PowerShard"
    OUTPUT_NODE = True
    def run(self, refresh):
        from .diagnostics import diagnose
        value = json.dumps(diagnose(), indent=2, ensure_ascii=False)
        return {"ui": {"text": [value]}, "result": (value,)}


class PowerShardSpectrum:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"model":("MODEL",),"enabled":("BOOLEAN",{"default":False}),
            "history_device":(["cpu","cuda"],),"history_mib":("INT",{"default":512,"min":0,"max":2147483647})},
            "optional":{"degree":("INT",{"default":2,"min":1,"max":4}),
                "warmup":("INT",{"default":3,"min":2,"max":2147483647}),
                "tail":("INT",{"default":1,"min":1,"max":2147483647}),
                "max_forecast":("INT",{"default":1,"min":1,"max":2147483647}),
                "history_size":("INT",{"default":6,"min":2,"max":2147483647}),
                "ridge":("FLOAT",{"default":.001,"min":1e-8,"max":100.}),
                "blend":("FLOAT",{"default":.5,"min":0.,"max":1.}),
                "audio_blend":("FLOAT",{"default":0.,"min":0.,"max":1.,"tooltip":"Спектральная доля audio. 0 оставляет линейный forecast; качество аудио требует сравнения."}),
                "allow_any_sampler":("BOOLEAN",{"default":False,"tooltip":"Opt-in: разрешить forecast для ЛЮБОГО sampler (не только deterministic Euler), s_churn=0. Для stochastic/multistage солверов качество не гарантировано — сравнивайте с disabled."})}}
    RETURN_TYPES=("MODEL",)
    FUNCTION="patch"
    CATEGORY="PowerShard/Patchers"
    def patch(self,model,enabled=False,history_device="cpu",history_mib=512,degree=2,warmup=3,tail=1,
              max_forecast=1,history_size=6,ridge=.001,blend=.5,audio_blend=0.,allow_any_sampler=False):
        from .comfy_adapter import PowerShardPatcher
        from .spectrum_config import SpectrumConfig
        if not isinstance(model,PowerShardPatcher):raise ValueError("Spectrum требует MODEL PowerShard H3 Loader")
        config=SpectrumConfig(enabled,degree,warmup,tail,max_forecast,history_size,history_mib,history_device,ridge,blend,audio_blend,allow_any_sampler)
        return (model.with_spectrum(config),)


NODE_CLASS_MAPPINGS = {"PowerShardConfig": PowerShardConfig,
                       "PowerShardConfigTuning": PowerShardConfigTuning,
                       "PowerShardH3MLP": PowerShardH3MLP, "PowerShardQwenMLP": PowerShardQwenMLP, "PowerShardH3Loader": PowerShardH3Loader,
                       "PowerShardH3QwenLoader":PowerShardH3QwenLoader,
                       "PowerShardSpectrum":PowerShardSpectrum,
                       "PowerShardH3FP16Patcher": PowerShardH3FP16Patcher,
                       "PowerShardH3TextEncoder": PowerShardH3TextEncoder, "PowerShardRelease": PowerShardRelease,
                       "PowerShardDiagnostics": PowerShardDiagnostics}
NODE_DISPLAY_NAME_MAPPINGS = {"PowerShardConfig": "PowerShard: GPU / память / attention",
    "PowerShardConfigTuning": "PowerShard: дополнительные настройки",
    "PowerShardH3MLP": "PowerShard: H3 MLP chunk", "PowerShardQwenMLP": "PowerShard: Qwen MLP chunk",
    "PowerShardSpectrum":"PowerShard Spectrum (приближённый, Euler)",
    "PowerShardH3QwenLoader":"PowerShard H3 Qwen Loader",
    "PowerShardH3FP16Patcher": "PowerShard MiniMax H3 FP16 Patcher",
    "PowerShardH3Loader": "PowerShard: H3 Sequence Loader",
    "PowerShardH3TextEncoder": "PowerShard: родной H3 Text Encoder", "PowerShardRelease": "PowerShard: освободить GPU перед VAE",
    "PowerShardDiagnostics": "PowerShard: диагностика среды"}
