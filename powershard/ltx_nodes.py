"""Ноды LTX-2 / 2.3 / 2.5 и ускорителей (Wan + LTX). Lazy imports: регистрация не импортирует torch/ComfyUI."""
from pathlib import Path
from .wan_nodes import _report_dir, _stat

CATEGORY = "PowerShard/LTX"
ACCEL_CATEGORY = "PowerShard/Accelerators"
FOLDERS = ("checkpoints", "diffusion_models")


def _model_choices():
    import folder_paths
    out = []
    for folder in FOLDERS:
        out += [f"{folder}/{name}" for name in folder_paths.get_filename_list(folder)]
    return out


def _resolve(choice):
    import folder_paths
    folder, _, name = choice.partition("/")
    if folder not in FOLDERS:
        raise ValueError(f"Ожидалось checkpoints/... или diffusion_models/..., получено {choice}")
    return Path(folder_paths.get_full_path_or_raise(folder, name))


def _choice_stat(choice):
    folder, _, name = choice.partition("/")
    return _stat(folder, name)


class PowerShardLTXOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fp16_safe": ("BOOLEAN", {"default": False, "tooltip": "Scaled FP16 GEMM (FP32 выход). LTX обучен в bf16: при NaN/Inf "
                                      "включите. Residual, AdaLN, RMSNorm, RoPE, timestep/caption/коннекторы — всегда FP32."}),
            "mlp_chunk_mode": (["off", "auto", "manual"], {"default": "auto", "tooltip": "FFN по частям токенов (auto — по свободной VRAM)."}),
            "mlp_chunk_tokens": ("INT", {"default": 4096, "min": 1, "max": 2147483647, "tooltip": "Только для mlp_chunk_mode=manual."}),
            "batch_chunk": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "0: весь CFG batch одним forward; 1: по одному."}),
            "attention_chunk": ("INT", {"default": 8192, "min": 256, "max": 1048576, "tooltip":
                                "Ключей на шаг частичной video->audio attention (FP32 online softmax)."})}}
    RETURN_TYPES = ("LTX_OPTIONS",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, fp16_safe=False, mlp_chunk_mode="auto", mlp_chunk_tokens=4096, batch_chunk=0, attention_chunk=8192):
        from .ltx_config import LTXOptions
        from .wan_nodes import debug_finite_from_env
        return (LTXOptions(fp16_safe=fp16_safe, debug_finite=debug_finite_from_env(), mlp_chunk_mode=mlp_chunk_mode,
                           mlp_chunk_tokens=mlp_chunk_tokens, batch_chunk=batch_chunk, attention_chunk=attention_chunk),)


class PowerShardLTXLoRA:
    """Устарело: используйте «PowerShard: LoRA» (тот же тип POWERSHARD_LORA). Оставлено для старых workflows."""
    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"lora_name": (folder_paths.get_filename_list("loras"),),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})},
                "optional": {"lora": ("POWERSHARD_LORA",)}}
    RETURN_TYPES = ("POWERSHARD_LORA",)
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(self, lora_name, strength=1.0, lora=None):
        import folder_paths
        from .wan_config import WanLoraSpec
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return (tuple(lora or ()) + (WanLoraSpec(str(path), float(strength), "all"),),)

    @classmethod
    def IS_CHANGED(cls, lora_name, strength=1.0, lora=None):
        return (_stat("loras", lora_name), strength, repr(lora))


class PowerShardLTXLoader:
    """LTX-2 / 2.3 / 2.5 (аудио+видео) или LTX-Video: FSDP2 + sequence parallel по видео-токенам."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"checkpoint": (_model_choices(), {"tooltip": "Полный checkpoint (checkpoints/) или только "
                                                                         "трансформер (diffusion_models/), bf16/fp16/fp8."}),
                             "config": ("POWERSHARD_CONFIG",)},
                "optional": {"keep_in_memory": ("BOOLEAN", {"default": True}),
                             "options": ("LTX_OPTIONS",), "weights": ("POWERSHARD_WEIGHTS",), "lora": ("POWERSHARD_LORA",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, checkpoint, config, keep_in_memory=True, options=None, weights=None, lora=None):
        from dataclasses import replace
        from .common_nodes import apply_weights
        from .ltx_adapter import load_ltx
        from .ltx_config import LTXOptions
        from .wan_nodes import debug_finite_from_env
        config = replace(config, release_after_sampling=not keep_in_memory)
        options = apply_weights(options or LTXOptions(debug_finite=debug_finite_from_env()), weights)
        return (load_ltx(str(_resolve(checkpoint)), config, options, tuple(lora or ()), _report_dir()),)

    @classmethod
    def IS_CHANGED(cls, checkpoint, config, keep_in_memory=True, options=None, weights=None, lora=None):
        from .web_api import provider_stamp
        return (_choice_stat(checkpoint), repr(config), keep_in_memory, repr(options), repr(weights), repr(lora),
                provider_stamp())


class PowerShardLTXVAELoader:
    """Видео- и аудио-VAE (+vocoder) из полного LTX checkpoint без чтения 40+ GB весов трансформера."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (_model_choices(),)}}
    RETURN_TYPES = ("VAE", "VAE")
    RETURN_NAMES = ("video_vae", "audio_vae")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, ckpt_name):
        import comfy.sd
        from safetensors import safe_open
        path = _resolve(ckpt_name)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            keys = list(f.keys())
            video = {k[len("vae."):]: f.get_tensor(k) for k in keys if k.startswith("vae.")}
            audio = {"autoencoder." + k[len("audio_vae."):]: f.get_tensor(k) for k in keys if k.startswith("audio_vae.")}
            audio.update({k: f.get_tensor(k) for k in keys if k.startswith("vocoder.")})
        if not video:
            raise ValueError(f"{path.name}: нет ключей vae.* (это не полный checkpoint) — используйте VAELoader")
        vae = comfy.sd.VAE(sd=video, metadata=metadata)
        vae.throw_exception_if_invalid()
        if not audio:
            raise ValueError(f"{path.name}: нет audio_vae.*/vocoder.* — используйте LTXVAudioVAELoader с отдельным файлом")
        audio_vae = comfy.sd.VAE(sd=audio, metadata=metadata)
        audio_vae.throw_exception_if_invalid()
        return (vae, audio_vae)

    @classmethod
    def IS_CHANGED(cls, ckpt_name):
        return _choice_stat(ckpt_name)


class PowerShardH3QuantLoader:
    """MiniMax H3 в пуле workers PowerShard с выбором хранения весов (узел «формат весов») и поддержкой Veda."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"checkpoint": (folder_paths.get_filename_list("diffusion_models"),),
                             "config": ("POWERSHARD_CONFIG",)},
                "optional": {"keep_in_memory": ("BOOLEAN", {"default": True}),
                             "weights": ("POWERSHARD_WEIGHTS", {"tooltip": "weight_format / compute / fp16_accumulation; "
                                                                           "weight_dtype для H3 не используется (всегда fp16)"}),
                             "lora": ("POWERSHARD_LORA", {"tooltip": "LoRA / LoHa / LoKr H3 (PTD, FastH3, PDD и др.): "
                                                                    "ключи разбирает ComfyUI, дельта сливается в shards"})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "PowerShard/MiniMax H3"

    def load(self, checkpoint, config, keep_in_memory=True, weights=None, lora=None):
        from dataclasses import replace
        import folder_paths
        from .common_nodes import weights_or_default
        from .h3_quant import load_h3
        w = weights_or_default(weights)
        path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint)
        config = replace(config, release_after_sampling=not keep_in_memory)
        return (load_h3(path, config, w.weight_format, w.compute, _report_dir(), bool(w.fp16_accumulation),
                        tuple(lora or ())),)

    @classmethod
    def IS_CHANGED(cls, checkpoint, config, keep_in_memory=True, weights=None, lora=None):
        from .web_api import provider_stamp
        return (_stat("diffusion_models", checkpoint), repr(config), keep_in_memory, repr(weights), repr(lora),
                provider_stamp())


# ------------------------------------------------------------------ accelerators
def _distributed(model):
    from .ltx_adapter import LTXPatcher
    from .wan_adapter import WanPatcher
    if not isinstance(model, (LTXPatcher, WanPatcher)):
        raise ValueError("Нужен MODEL из PowerShard Wan/LTX loader (для native моделей используйте EasyCache/TeaCache)")
    return model


def _sigma(model, percent):
    return float(model.get_model_object("model_sampling").percent_to_sigma(percent))


class PowerShardBlockCache:
    """TeaCache/FBCache-подобный пропуск блоков в workers: блок 0 считается всегда, остальные — когда меняются."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "threshold": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005, "tooltip":
                                           "Относительное L1-изменение residual блока 0 (все rank, all-reduce). 0.05–0.12; больше — "
                                           "быстрее и грубее."}),
                             "start_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "max_consecutive_skips": ("INT", {"default": 3, "min": 1, "max": 100}),
                             "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 100})},
                "optional": {"max_cache_mb": ("INT", {"default": 1024, "min": 64, "max": 65536, "step": 64, "tooltip":
                                              "Предел VRAM на кэш residual в каждом GPU (LRU). Одна запись Wan 14B 480p/81 кадр "
                                              "≈ 450 МБ; с Context Windows нужна запись на окно."})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = ACCEL_CATEGORY

    def apply(self, model, threshold=0.08, start_percent=0.15, end_percent=0.95, max_consecutive_skips=3, warmup_steps=2,
              max_cache_mb=1024):
        from .accel import BlockCacheHolder, install_holder
        model = _distributed(model)
        holder = BlockCacheHolder(threshold, _sigma(model, start_percent), _sigma(model, end_percent),
                                  max_consecutive_skips, warmup_steps, max_cache_mb)
        return (install_holder(model, "powershard_block_cache", holder),)


class PowerShardNAG:
    """Normalized Attention Guidance: негативный prompt без CFG (distilled/Lightning/LTX distilled при cfg=1)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "negative": ("CONDITIONING",),
                             "nag_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 50.0, "step": 0.1}),
                             "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.05}),
                             "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "apply_to_audio": ("BOOLEAN", {"default": True, "tooltip": "LTX-AV: и в аудио cross-attention."})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = ACCEL_CATEGORY

    def apply(self, model, negative, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25, start_percent=0.0, end_percent=1.0,
              apply_to_audio=True):
        from .accel import NAGHolder, install_holder
        model = _distributed(model)
        if not negative:
            raise ValueError("NAG: пустой negative conditioning")
        context, extras = negative[0][0], negative[0][1]
        holder = NAGHolder(context.detach().float().cpu(), nag_scale, nag_tau, nag_alpha, _sigma(model, start_percent),
                           _sigma(model, end_percent), apply_to_audio, extras.get("unprocessed_ltxav_embeds", False))
        return (install_holder(model, "powershard_nag", holder),)


class PowerShardRIFLEx:
    """RIFLEx (Wan): длиннее обучающих 81 кадра без повторов движения — одна временная частота RoPE."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "k": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Индекс частоты (1-based); 0 — авто по длине обучения."}),
                             "train_latent_frames": ("INT", {"default": 21, "min": 1, "max": 1024, "tooltip":
                                                     "Длина обучения в latent-кадрах (Wan: 81 кадр -> 21)."})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = ACCEL_CATEGORY

    def apply(self, model, k=0, train_latent_frames=21):
        from .accel import RIFLExHolder, install_holder
        from .wan_adapter import WanPatcher
        if not isinstance(model, WanPatcher):
            raise ValueError("RIFLEx реализован для PowerShard Wan (RoPE Wan); у LTX дробные позиции RoPE")
        return (install_holder(model, "powershard_riflex", RIFLExHolder(k, train_latent_frames)),)


LTX_NODE_CLASS_MAPPINGS = {
    "PowerShardLTXOptions": PowerShardLTXOptions,
    "PowerShardLTXLoRA": PowerShardLTXLoRA,
    "PowerShardLTXLoader": PowerShardLTXLoader,
    "PowerShardLTXVAELoader": PowerShardLTXVAELoader,
    "PowerShardH3QuantLoader": PowerShardH3QuantLoader,
    "PowerShardBlockCache": PowerShardBlockCache,
    "PowerShardNAG": PowerShardNAG,
    "PowerShardRIFLEx": PowerShardRIFLEx,
}
LTX_NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerShardLTXOptions": "PowerShard LTX: численная политика / память",
    "PowerShardLTXLoRA": "PowerShard LTX: LoRA / IC-LoRA (слияние в shards)",
    "PowerShardLTXLoader": "PowerShard LTX-2/2.5: Sequence Loader",
    "PowerShardLTXVAELoader": "PowerShard LTX: Video+Audio VAE из checkpoint (без DiT)",
    "PowerShardH3QuantLoader": "PowerShard MiniMax H3: Loader",
    "PowerShardBlockCache": "PowerShard: Block Cache (TeaCache/FBCache, Wan+LTX)",
    "PowerShardNAG": "PowerShard: NAG (Normalized Attention Guidance)",
    "PowerShardRIFLEx": "PowerShard Wan: RIFLEx (длинное видео)",
}
