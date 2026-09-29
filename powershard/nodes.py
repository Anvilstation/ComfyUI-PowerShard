"""Lazy imports позволяют проверить регистрацию нод без torch/CUDA/ComfyUI."""
import json
from pathlib import Path
from .config import DistributedConfig


class PowerShardConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"gpu_ids": ("STRING", {"default": "0,1,2", "tooltip": "CUDA-visible индексы ComfyUI, порядок сохраняется. Например 5,2,0 или all."}),
            "backend": (["fsdp2", "fsdp2_sequence"],), "precision": (["fp16", "int8_fp16"],),
            "reserve_gib": ("FLOAT", {"default": 2., "min": 0., "max": 1e9}),
            "timeout_s": ("INT", {"default": 600, "min": 1, "max": 2147483647}),
            "allow_unverified": ("BOOLEAN", {"default": False, "tooltip": "Устаревший compatibility field; больше не блокирует запуск. Допуск по capabilities/preflight."}),
            "release_after_sampling": ("BOOLEAN", {"default": True})},
            "optional": {"cpu_offload": ("BOOLEAN", {"default": False}),
                         "pin_memory": ("BOOLEAN", {"default": True}),
                         "prefetch_blocks": ([0, 1, 2], {"default": 0}),
                         "numa_policy": (["none", "auto", "bind"],),
                         "attention_backend": (["auto", "sdpa", "flash_attn", "vllm_flash_attn", "sageattention", "math"], {"default":"math", "tooltip":"sdpa: PyTorch SDPA; flash_attn: установленный FlashAttention; vllm_flash_attn: custom/vLLM kernels; sageattention: квантованный opt-in; math: reference. auto не означает fastest."}),
                         "allow_fallback": ("BOOLEAN", {"default":True}),
                         "memory_policy": (["manual", "auto"], {"default":"manual"}),
                         "weight_placement": (["gpu", "cpu", "ats"], {"default": "gpu", "tooltip": "gpu: учитывает legacy cpu_offload. cpu: CPUOffloadPolicy. ats: тот же offload + изолированная диагностика; отдельный ATS allocation path не реализован."}),
                         "sequence_mode": (["token", "ulysses"], {"default":"token", "tooltip":"token: разрез по токенам, full K/V gather (3 GPU). ulysses: разрез по heads через all-to-all, требует heads % world == 0 (H3: 2/4/7/8/14/28), иначе откат в token."}),
                         "sequence_comm_dtype": (["fp16", "fp32"], {"default":"fp32"}),
                         "memory_profile": (["custom", "ram_min"], {"default":"custom"}),
                         "workspace_mib": ("INT", {"default":256, "min":1, "max":2147483647}),
                         "stage_cache_mib": ("INT", {"default":64, "min":0, "max":2147483647})}}
    RETURN_TYPES = ("POWERSHARD_CONFIG",)
    FUNCTION = "create"
    CATEGORY = "PowerShard"
    def create(self, gpu_ids, backend, precision, reserve_gib, timeout_s, allow_unverified, release_after_sampling,
               cpu_offload=False, pin_memory=True, prefetch_blocks=0, numa_policy="none", attention_backend=None, allow_fallback=True, memory_policy="manual",
               weight_placement="gpu", sequence_mode="token", sequence_comm_dtype="fp32",
               memory_profile="custom", workspace_mib=256, stage_cache_mib=64):
        return (DistributedConfig(tuple(gpu_ids.split(",")), backend=backend, precision=precision,
                                  reserve_gib=reserve_gib, timeout_s=timeout_s, allow_unverified=allow_unverified,
                                  release_after_sampling=release_after_sampling, cpu_offload=cpu_offload,
                                  pin_memory=pin_memory, prefetch_blocks=int(prefetch_blocks), numa_policy=numa_policy,
                                  attention_backend=attention_backend,allow_fallback=allow_fallback,memory_policy=memory_policy,
                                  weight_placement=weight_placement, sequence_mode=sequence_mode,
                                  sequence_comm_dtype=sequence_comm_dtype, memory_profile=memory_profile,
                                  workspace_mib=workspace_mib, stage_cache_mib=stage_cache_mib),)


class PowerShardH3Loader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"checkpoint": (folder_paths.get_filename_list("diffusion_models"),), "config": ("POWERSHARD_CONFIG",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "PowerShard"
    def load(self, checkpoint, config):
        import folder_paths
        from .comfy_adapter import load_model
        path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint)
        return (load_model(path, config, Path(folder_paths.get_output_directory())/"powershard"),)
    @classmethod
    def IS_CHANGED(cls, checkpoint, config):
        import folder_paths
        p = Path(folder_paths.get_full_path_or_raise("diffusion_models", checkpoint))
        st = p.stat()
        from .web_api import provider_stamp
        return (st.st_size, st.st_mtime_ns, repr(config), provider_stamp())


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
            "clear_conditioning_cache":("BOOLEAN",{"default":False})}}
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "report")
    FUNCTION = "release"
    CATEGORY = "PowerShard"
    def release(self, samples, preserve_qwen_cpu_shards=True, clear_conditioning_cache=False):
        from .runtime import release_all
        release_all(preserve_idle_cpu=preserve_qwen_cpu_shards)
        if clear_conditioning_cache:
            from .conditioning_cache import clear_all_caches
            clear_all_caches()
        return (samples, "Активные PowerShard workers завершены. Idle Qwen CPU shards сохранены: "+str(preserve_qwen_cpu_shards))


class PowerShardH3QwenLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required":{"checkpoint":(folder_paths.get_filename_list("text_encoders"),),
            "config":("POWERSHARD_CONFIG",),"precision":(["fp16","int8_fp16"],),
            "idle_policy":(["release","cpu_shards","keep"],),
            "cache_mib":("INT",{"default":256,"min":0,"max":2147483647})},"optional":{
                "mlp_chunk_mode":(["auto","manual","off"],),
                "mlp_chunk_tokens":("INT",{"default":4096,"min":1,"max":2147483647})}}
    RETURN_TYPES=("CLIP",)
    FUNCTION="load"
    CATEGORY="PowerShard"
    def load(self,checkpoint,config,precision="fp16",idle_policy="release",cache_mib=256,mlp_chunk_mode="auto",mlp_chunk_tokens=4096):
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
        return {"required": {"model": ("MODEL",), "enabled": ("BOOLEAN", {"default": True}),
                             "fp16_safe": ("BOOLEAN", {"default": True}),
                             "debug_finite": ("BOOLEAN", {"default": False})},
                "optional": {"mlp_chunk_tokens": ("INT", {"default": 512, "min": 1, "max": 2147483647}),
                             "mlp_chunk_mode": (["manual", "auto", "off"], {"default":"manual", "tooltip":"off отключает только деление MLP; FP16 Safe остаётся включённым. Старые workflows = manual."})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "PowerShard/Patchers"

    def patch(self, model, enabled=True, fp16_safe=True, debug_finite=False, mlp_chunk_tokens=512, mlp_chunk_mode="manual"):
        from .comfy_adapter import PowerShardPatcher
        from .patch_config import H3PatchConfig
        if not isinstance(model, PowerShardPatcher):
            raise ValueError("Нужен MODEL из PowerShard H3 Model Loader; native локальную H3 этот distributed patcher не изменяет")
        return (model.with_h3_patch(H3PatchConfig(enabled, fp16_safe, debug_finite, mlp_chunk_tokens,mlp_chunk_mode)),)


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


NODE_CLASS_MAPPINGS = {"PowerShardConfig": PowerShardConfig, "PowerShardH3Loader": PowerShardH3Loader,
                       "PowerShardH3QwenLoader":PowerShardH3QwenLoader,
                       "PowerShardSpectrum":PowerShardSpectrum,
                       "PowerShardH3FP16Patcher": PowerShardH3FP16Patcher,
                       "PowerShardH3TextEncoder": PowerShardH3TextEncoder, "PowerShardRelease": PowerShardRelease,
                       "PowerShardDiagnostics": PowerShardDiagnostics}
NODE_DISPLAY_NAME_MAPPINGS = {"PowerShardConfig": "PowerShard: распределённая конфигурация",
    "PowerShardSpectrum":"PowerShard Spectrum (приближённый, Euler)",
    "PowerShardH3QwenLoader":"PowerShard H3 Qwen Loader",
    "PowerShardH3FP16Patcher": "PowerShard MiniMax H3 FP16 Patcher",
    "PowerShardH3Loader": "PowerShard: H3 FSDP Loader (экспериментальный)",
    "PowerShardH3TextEncoder": "PowerShard: родной H3 Text Encoder", "PowerShardRelease": "PowerShard: освободить GPU перед VAE",
    "PowerShardDiagnostics": "PowerShard: диагностика среды"}

# Только собственные ноды: порядок INPUT_TYPES, имена widgets и class IDs прежние.
from .ui_schema import annotate_nodes
annotate_nodes(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
