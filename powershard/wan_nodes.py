"""Ноды Wan 2.1/2.2. Lazy imports: регистрация не импортирует torch/CUDA/ComfyUI."""
from pathlib import Path

CATEGORY = "PowerShard/Wan"
MAX_RESOLUTION = 16384  # = comfy nodes.MAX_RESOLUTION (без импорта nodes/torch при регистрации)


def _report_dir():
    import folder_paths
    return Path(folder_paths.get_output_directory()) / "powershard"


def _stat(folder, name):
    import folder_paths
    p = Path(folder_paths.get_full_path_or_raise(folder, name))
    st = p.stat()
    return (str(p), st.st_size, st.st_mtime_ns)


class PowerShardWanOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fp16_safe": ("BOOLEAN", {"default": False, "tooltip": "Scaled FP16 GEMM (power-of-two, FP32 выход) как H3 Safe. "
                                      "По умолчанию обычный FP16 GEMM: Wan рассчитан на fp16 в ComfyUI; residual/модуляция/нормы всё равно FP32. "
                                      "Включайте при NaN/Inf."}),
            "debug_finite": ("BOOLEAN", {"default": False}),
            "mlp_chunk_mode": (["off", "auto", "manual"], {"default": "off", "tooltip": "FFN по частям токенов. auto — по свободной VRAM. Для 720p/81 кадров на 16 GB рекомендован auto."}),
            "mlp_chunk_tokens": ("INT", {"default": 4096, "min": 1, "max": 2147483647}),
            "moe_residency": (["swap", "both"], {"default": "swap", "tooltip": "swap: неактивный эксперт паркуется в RAM при переключении high->low "
                                                 "(раз за прогон; при placement=cpu это лишь reshard). both: оба эксперта активны — при placement=gpu "
                                                 "это ~9.5 GB весов на каждую V100 и мало места для активаций."}),
            "batch_chunk": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "0: CFG cond+uncond одним forward. 1: последовательно внутри worker (меньше активаций, веса собираются дважды)."}),
            "model_type": (["auto", "animate2"], {"default": "auto", "tooltip": "auto — по весам/metadata checkpoint. animate2 — Wan-Animate-2: "
                           "его checkpoint имеет форму Wan2.1 I2V и без metadata неотличим по весам."})},
                "optional": {"weight_format": (["dequantize", "as_file", "int8", "fp8", "mxfp8", "nvfp4", "int4"], {"default": "dequantize", "tooltip":
                             "Как хранить Linear-веса блоков в VRAM/RAM. dequantize — fp16/bf16 (как раньше). as_file — в формате "
                             "файла (int8/fp8/nvfp4/mxfp8/int4 остаются упакованными, bf16-слои — bf16). int8/fp8/mxfp8/nvfp4/int4 — "
                             "квантовать при загрузке (bf16/fp16 файл тоже). Экономит VRAM и трафик FSDP в 2–4 раза."}),
                             "compute": (["auto", "native", "dequantize"], {"default": "auto", "tooltip":
                             "Как считать квантованные слои. native — низкобитные ядра comfy-kitchen (int8/int4: sm75+, fp8: sm89+, "
                             "nvfp4/mxfp8: sm100+/RTX 50; CUDA backend comfy-kitchen = torch cu130+). dequantize — вес распаковывается "
                             "в fp16/bf16 на время GEMM (работает везде, в т.ч. V100). auto — native, если GPU и backend умеют формат."}),
                             "weight_dtype": (["auto", "fp16", "bf16"], {"default": "auto", "tooltip": "Хранение/GEMM весов в workers. auto: bf16 на "
                             "RTX 30/40/50, A100/H100 (sm80+), fp16 на V100/T4. Любой формат файла (bf16/fp16/fp8/int8/nvfp4/"
                             "mxfp8/int4) деквантуется в этот dtype по локальным строкам."})}}
    RETURN_TYPES = ("WAN_OPTIONS",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, fp16_safe=False, debug_finite=False, mlp_chunk_mode="off", mlp_chunk_tokens=4096,
               moe_residency="swap", batch_chunk=0, model_type="auto", weight_dtype="auto", weight_format="dequantize",
               compute="auto"):
        from .wan_config import WanOptions
        return (WanOptions(fp16_safe, debug_finite, mlp_chunk_mode, mlp_chunk_tokens, moe_residency, batch_chunk,
                           model_type, weight_dtype, weight_format, compute),)


class PowerShardWanLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"lora_name": (folder_paths.get_filename_list("loras"),),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "apply_to": (["all", "high_noise", "low_noise"], {"default": "all",
                                          "tooltip": "Для MoE A14B: к какому эксперту. Lightning/lightx2v обычно имеют отдельные high/low файлы."})},
                "optional": {"lora": ("WAN_LORA",)}}
    RETURN_TYPES = ("WAN_LORA",)
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(self, lora_name, strength=1.0, apply_to="all", lora=None):
        import folder_paths
        from .wan_config import WanLoraSpec
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return (tuple(lora or ()) + (WanLoraSpec(str(path), float(strength), apply_to),),)

    @classmethod
    def IS_CHANGED(cls, lora_name, strength=1.0, apply_to="all", lora=None):
        return (_stat("loras", lora_name), strength, apply_to, repr(lora))


def _options(options):
    from .wan_config import WanOptions
    return options if options is not None else WanOptions()


class PowerShardWanLoader:
    """Одна модель: Wan 2.2 TI2V-5B, Wan 2.1 T2V/I2V/FLF или отдельный эксперт A14B."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"checkpoint": (folder_paths.get_filename_list("diffusion_models"),),
                             "config": ("POWERSHARD_CONFIG",)},
                "optional": {"keep_in_memory": ("BOOLEAN", {"default": True}),
                             "options": ("WAN_OPTIONS",), "lora": ("WAN_LORA",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, checkpoint, config, keep_in_memory=True, options=None, lora=None):
        from dataclasses import replace
        import folder_paths
        from .wan_adapter import load_wan
        path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint)
        config = replace(config, release_after_sampling=not keep_in_memory)
        base = load_wan({"main": str(path)}, config, _options(options), tuple(lora or ()), _report_dir())
        return (base.with_expert("main"),)

    @classmethod
    def IS_CHANGED(cls, checkpoint, config, keep_in_memory=True, options=None, lora=None):
        from .web_api import provider_stamp
        return (_stat("diffusion_models", checkpoint), repr(config), keep_in_memory, repr(options), repr(lora), provider_stamp())


class PowerShardWan22MoELoader:
    """Wan 2.2 A14B: high-noise и low-noise эксперты в ОДНОМ пуле workers."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        models = folder_paths.get_filename_list("diffusion_models")
        return {"required": {"high_noise_checkpoint": (models,), "low_noise_checkpoint": (models,),
                             "config": ("POWERSHARD_CONFIG",),
                             "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.005,
                                          "tooltip": "Только для выхода moe: high пока timestep >= boundary*1000. Wan 2.2: T2V 0.875, I2V 0.900."})},
                "optional": {"keep_in_memory": ("BOOLEAN", {"default": True}),
                             "options": ("WAN_OPTIONS",), "lora": ("WAN_LORA",)}}
    RETURN_TYPES = ("MODEL", "MODEL", "MODEL")
    RETURN_NAMES = ("high_noise", "low_noise", "moe")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, high_noise_checkpoint, low_noise_checkpoint, config, boundary=0.875, keep_in_memory=True,
             options=None, lora=None):
        from dataclasses import replace
        import folder_paths
        from .wan_adapter import load_wan
        if high_noise_checkpoint == low_noise_checkpoint:
            raise ValueError("high_noise и low_noise должны быть разными checkpoint")
        experts = {"high": str(folder_paths.get_full_path_or_raise("diffusion_models", high_noise_checkpoint)),
                   "low": str(folder_paths.get_full_path_or_raise("diffusion_models", low_noise_checkpoint))}
        config = replace(config, release_after_sampling=not keep_in_memory)
        base = load_wan(experts, config, _options(options), tuple(lora or ()), _report_dir(), boundary)
        return (base.with_expert("high"), base.with_expert("low"), base.with_expert("moe", boundary))

    @classmethod
    def IS_CHANGED(cls, high_noise_checkpoint, low_noise_checkpoint, config, boundary=0.875, keep_in_memory=True,
                   options=None, lora=None):
        from .web_api import provider_stamp
        return (_stat("diffusion_models", high_noise_checkpoint), _stat("diffusion_models", low_noise_checkpoint),
                repr(config), boundary, keep_in_memory, repr(options), repr(lora), provider_stamp())


class PowerShardWanTextEncoder:
    """umT5-XXL для Wan. cpu_fp32 — точный FP32 на POWER9 (≈23 GB RAM), не занимает V100."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"checkpoint": (folder_paths.get_filename_list("text_encoders"),),
                             "placement": (["cpu_fp32", "native_offload"], {"default": "cpu_fp32"})}}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, checkpoint, placement="cpu_fp32"):
        import torch
        import comfy.sd
        import folder_paths
        options = ({"load_device": torch.device("cpu"), "offload_device": torch.device("cpu"), "dtype": torch.float32}
                   if placement == "cpu_fp32" else {})
        clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", checkpoint)],
                                  embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                  clip_type=comfy.sd.CLIPType.WAN, model_options=options)
        return (clip,)


class PowerShardWanInfo:
    """Геометрия/формат checkpoint без загрузки весов: dim, heads, семейство, fp8/bf16."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"checkpoint": (folder_paths.get_filename_list("diffusion_models"),),
                             "gpu_count": ("INT", {"default": 6, "min": 1, "max": 64})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def inspect(self, checkpoint, gpu_count=6):
        import json
        import folder_paths
        from .wan_config import WanCheckpoint, describe_family, memory_plan
        ckpt = WanCheckpoint(folder_paths.get_full_path_or_raise("diffusion_models", checkpoint))
        geometry = ckpt.model_config()
        value = json.dumps(dict(family=describe_family(geometry), geometry=geometry, storage=ckpt.storage(),
                                key_prefix=ckpt.prefix, memory=memory_plan(ckpt, gpu_count)),
                           indent=2, ensure_ascii=False, default=str)
        return {"ui": {"text": [value]}, "result": (value,)}


class PowerShardWanUni3C:
    """ControlNet для Wan в ComfyUI = Uni3C (камера/рендер облака точек). Веса грузятся в workers."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"model": ("MODEL",),
                             "controlnet": (folder_paths.get_filename_list("model_patches"),),
                             "vae": ("VAE",), "render_video": ("IMAGE",), "latent": ("LATENT", {"tooltip": "Латент, который пойдёт в sampler: задаёт кадры/размер render latent."}),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, controlnet, vae, render_video, latent, strength=1.0, start_percent=0.0, end_percent=1.0):
        import comfy.utils
        import folder_paths
        from .wan_adapter import WanPatcher
        from .wan_config import Uni3CCheckpoint
        if not isinstance(model, WanPatcher):
            raise ValueError("Нужен MODEL из PowerShard Wan loader")
        path = Path(folder_paths.get_full_path_or_raise("model_patches", controlnet))
        geometry = Uni3CCheckpoint(path).model_config()
        dim = model.model.diffusion_model.dim
        if geometry["out_proj_dim"] != dim:
            raise ValueError(f"Uni3C ожидает Wan dim {geometry['out_proj_dim']}, загружен dim {dim}")
        t_len, h_len, w_len = latent["samples"].shape[-3:]
        frames = render_video[:, :, :, :3]
        target = (t_len - 1) * (vae.temporal_compression_decode() or 1) + 1
        if frames.shape[0] > target:
            frames = frames[:target]
        elif frames.shape[0] < target:
            frames = torch_cat_last(frames, target - frames.shape[0])
        spatial = vae.spacial_compression_encode()
        if frames.shape[1] != h_len * spatial or frames.shape[2] != w_len * spatial:
            frames = comfy.utils.common_upscale(frames.movedim(-1, 1), w_len * spatial, h_len * spatial,
                                                "bilinear", "center").movedim(1, -1)
        render = model.get_model_object("latent_format").process_in(vae.encode(frames)).float().cpu()
        sampling = model.get_model_object("model_sampling")
        st = path.stat()
        return (model.with_uni3c(dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns, strength=float(strength),
                                      sigma_start=float(sampling.percent_to_sigma(start_percent)),
                                      sigma_end=float(sampling.percent_to_sigma(end_percent)), render=render)),)


def load_multitalk_audio_proj(path):
    """audio_proj из InfiniteTalk/MultiTalk patch на host (CPU FP32): ~50M параметров, нужен один раз на ноду."""
    import torch
    import comfy.ops
    from safetensors import safe_open
    from comfy.ldm.wan.model_multitalk import MultiTalkAudioProjModel
    with safe_open(str(path), framework="pt", device="cpu") as f:
        state = {k[len("audio_proj."):]: f.get_tensor(k).float() for k in f.keys() if k.startswith("audio_proj.")}
    if "proj1.weight" not in state:
        raise ValueError(f"{Path(path).name}: это не InfiniteTalk/MultiTalk model patch (нет audio_proj.proj1)")
    proj = MultiTalkAudioProjModel(seq_len=5, seq_len_vf=5 + 4 - 1, intermediate_dim=state["proj1.weight"].shape[0],
                                   out_dim=state["norm.weight"].shape[0], context_tokens=32, dtype=torch.float32,
                                   device="cpu", operations=comfy.ops.manual_cast)  # как ModelPatchLoader: веса к device/dtype входа
    proj.load_state_dict(state, strict=True)
    return proj.eval().requires_grad_(False)


class PowerShardWanInfiniteTalk:
    """InfiniteTalk / MultiTalk (1-2 говорящих) для PowerShard Wan.

    Логика native WanInfiniteTalkToVideo (кадры, аудио-окна, motion frames, маски) выполняется
    родным кодом comfy; 40 audio cross-attention блоков model patch грузятся в workers (FSDP,
    свои локальные строки), audio_proj — на host.
    """
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        size = dict(min=16, max=MAX_RESOLUTION, step=16)
        return {"required": {"mode": (["single_speaker", "two_speakers"], {"default": "single_speaker"}),
                             "model": ("MODEL",),
                             "model_patch": (folder_paths.get_filename_list("model_patches"),
                                             {"tooltip": "Тот же файл, что для native ModelPatchLoader (InfiniteTalk single/multi)."}),
                             "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "vae": ("VAE",),
                             "width": ("INT", dict(size, default=832)), "height": ("INT", dict(size, default=480)),
                             "length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 4}),
                             "audio_encoder_output_1": ("AUDIO_ENCODER_OUTPUT",),
                             "motion_frame_count": ("INT", {"default": 9, "min": 1, "max": 33, "step": 1}),
                             "audio_scale": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})},
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT",), "start_image": ("IMAGE",),
                             "previous_frames": ("IMAGE",), "audio_encoder_output_2": ("AUDIO_ENCODER_OUTPUT",),
                             "mask_1": ("MASK",), "mask_2": ("MASK",)}}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "trim_image")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    _proj_cache = {}

    def apply(self, mode, model, model_patch, positive, negative, vae, width, height, length, audio_encoder_output_1,
              motion_frame_count=9, audio_scale=1.0, clip_vision_output=None, start_image=None, previous_frames=None,
              audio_encoder_output_2=None, mask_1=None, mask_2=None):
        import types
        import folder_paths
        from comfy_extras.nodes_wan import WanInfiniteTalkToVideo
        from .wan_adapter import WanPatcher
        from .wan_config import MultiTalkCheckpoint
        if not isinstance(model, WanPatcher):
            raise ValueError("Нужен MODEL из PowerShard Wan loader (Wan 2.1 I2V 14B)")
        path = Path(folder_paths.get_full_path_or_raise("model_patches", model_patch))
        geometry = MultiTalkCheckpoint(path).model_config()
        proxy = model.model.diffusion_model
        if geometry["in_dim"] != proxy.dim or geometry["num_layers"] != proxy.geometry["num_layers"]:
            raise ValueError(f"InfiniteTalk patch: dim {geometry['in_dim']} / {geometry['num_layers']} блоков, "
                             f"модель: dim {proxy.dim} / {proxy.geometry['num_layers']} блоков")
        st = path.stat()
        stamp = (str(path), st.st_size, st.st_mtime_ns)
        if stamp not in self._proj_cache:
            self._proj_cache.clear()
            self._proj_cache[stamp] = load_multitalk_audio_proj(path)
        standin = types.SimpleNamespace(model=types.SimpleNamespace(audio_proj=self._proj_cache[stamp]))
        if mode == "single_speaker":
            audio_encoder_output_2 = mask_1 = mask_2 = None
        values = {"mode": mode, "audio_encoder_output_2": audio_encoder_output_2, "mask_1": mask_1, "mask_2": mask_2}
        output = WanInfiniteTalkToVideo.execute(values, model, standin, positive, negative, vae, width, height, length,
                                                audio_encoder_output_1, motion_frame_count, start_image=start_image,
                                                previous_frames=previous_frames, audio_scale=audio_scale,
                                                clip_vision_output=clip_vision_output,
                                                audio_encoder_output_2=audio_encoder_output_2, mask_1=mask_1, mask_2=mask_2)
        patched, positive, negative, latent, trim = output.args
        options = patched.model_options.setdefault("transformer_options", {})
        audio = options.pop("audio_embeds")
        masks, scale = None, float(audio_scale)
        patches = {}
        for name, group in (options.get("patches") or {}).items():
            kept = []
            for p in group:
                if type(p).__name__ == "MultiTalkCrossAttnPatch" and p.model_patch is standin:
                    scale = float(p.audio_scale)
                elif type(p).__name__ == "MultiTalkGetAttnMapPatch":
                    masks = p.ref_target_masks
                else:
                    kept.append(p)
            if kept:
                patches[name] = kept
        options["patches"] = patches
        spec = dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns, audio_scale=scale,
                    audio=audio.detach().float().cpu(), masks=None if masks is None else masks.detach().float().cpu())
        return (patched.with_multitalk(spec), positive, negative, latent, trim)

    @classmethod
    def IS_CHANGED(cls, model_patch, **kwargs):
        return _stat("model_patches", model_patch)


class PowerShardWanT5Distributed:
    """umT5-XXL в пуле PowerShard workers: веса шардируются по GPU выбранного config, а не ложатся на cuda:0."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"checkpoint": (folder_paths.get_filename_list("text_encoders"),),
                             "config": ("POWERSHARD_CONFIG", {"tooltip": "Тот же PowerShardConfig, что у генератора (GPU, placement). "
                                                                         "precision для энкодера всегда fp16 storage / FP32 вычисление."}),
                             "after_encode": (["release", "release_now", "keep_ram"], {"default": "release", "tooltip":
                                              "release: workers энкодера живут, пока кодируются промпты этого запуска, и закрываются при "
                                              "старте генератора PowerShard / Free VRAM. release_now: закрыть сразу после каждого encode. "
                                              "keep_ram: shards остаются в pinned RAM (weight_placement=cpu). Одинаковый текст всегда "
                                              "берётся из кэша без запуска workers."})}}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, checkpoint, config, after_encode="release"):
        import folder_paths
        from .wan_text import load_wan_t5
        path = folder_paths.get_full_path_or_raise("text_encoders", checkpoint)
        return (load_wan_t5(path, config, after_encode, _report_dir(), folder_paths.get_folder_paths("embeddings")),)

    @classmethod
    def IS_CHANGED(cls, checkpoint, config, after_encode="release"):
        return (_stat("text_encoders", checkpoint), repr(config), after_encode)


class PowerShardFreeVRAM:
    """Выгрузить host-модели (text encoder, CLIP vision, VAE) из VRAM перед sampling.

    ComfyUI держит их на cuda:0, т.к. PowerShard-генератор почти не занимает host VRAM; а cuda:0 —
    это ещё и rank 0 workers. Нода пропускает conditioning насквозь: поставьте её между
    encode/…ToVideo и sampler, чтобы выгрузка произошла до первого шага.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                             "mode": (["selected", "all_native"], {"default": "selected", "tooltip":
                                      "selected: только подключённые clip/clip_vision/vae. all_native: все native модели ComfyUI "
                                      "(PowerShard-модели не трогаются)."})},
                "optional": {"clip": ("CLIP",), "clip_vision": ("CLIP_VISION",), "vae": ("VAE",), "latent": ("LATENT",)}}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "report")
    FUNCTION = "free"
    CATEGORY = CATEGORY

    def free(self, positive, negative, mode="selected", clip=None, clip_vision=None, vae=None, latent=None):
        import json
        from .wan_free import free_host_models
        report = free_host_models(mode, [m for m in (clip, clip_vision, vae) if m is not None])
        return (positive, negative, latent, json.dumps(report, ensure_ascii=False, indent=1))



class PowerShardWanAnimate2Cache:
    """Кэш pose branch Animate2 в RAM workers (аналог WanAnimate2Cache без host-объектов и callbacks)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "dtype": (["fp16", "fp32"], {"default": "fp16", "tooltip": "Хранение входов pose branch в RAM workers. "
                                       "fp16 ≈ 2.3 GB на GPU-rank при 480p/81 кадр (6 GPU), fp32 — вдвое больше."})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, dtype="fp16"):
        import uuid
        from .wan_adapter import WanPatcher, ANIMATE2_CACHE_KEY
        if not isinstance(model, WanPatcher):
            raise ValueError("Нужен MODEL из PowerShard Wan loader (Animate2)")
        result = model.clone()
        result.model_options.setdefault("transformer_options", {})[ANIMATE2_CACHE_KEY] = dict(id=uuid.uuid4().hex, dtype=dtype)
        return (result,)


def torch_cat_last(frames, count):
    import torch
    return torch.cat([frames, frames[-1:].expand(count, -1, -1, -1)], dim=0)


WAN_NODE_CLASS_MAPPINGS = {
    "PowerShardWanOptions": PowerShardWanOptions,
    "PowerShardWanLoRA": PowerShardWanLoRA,
    "PowerShardWanLoader": PowerShardWanLoader,
    "PowerShardWan22MoELoader": PowerShardWan22MoELoader,
    "PowerShardWanTextEncoder": PowerShardWanTextEncoder,
    "PowerShardWanInfo": PowerShardWanInfo,
    "PowerShardWanUni3C": PowerShardWanUni3C,
    "PowerShardWanInfiniteTalk": PowerShardWanInfiniteTalk,
    "PowerShardWanT5Distributed": PowerShardWanT5Distributed,
    "PowerShardFreeVRAM": PowerShardFreeVRAM,
    "PowerShardWanAnimate2Cache": PowerShardWanAnimate2Cache,
}
WAN_NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerShardWanOptions": "PowerShard Wan: численная политика / память",
    "PowerShardWanLoRA": "PowerShard Wan: LoRA (слияние в shards)",
    "PowerShardWanLoader": "PowerShard Wan: Sequence Loader (одна модель)",
    "PowerShardWan22MoELoader": "PowerShard Wan 2.2: MoE Loader (high+low)",
    "PowerShardWanTextEncoder": "PowerShard Wan: umT5 Text Encoder",
    "PowerShardWanInfo": "PowerShard Wan: checkpoint info",
    "PowerShardWanUni3C": "PowerShard Wan: Uni3C ControlNet",
    "PowerShardWanInfiniteTalk": "PowerShard Wan: InfiniteTalk / MultiTalk",
    "PowerShardWanT5Distributed": "PowerShard Wan: umT5 Distributed Encoder",
    "PowerShardFreeVRAM": "PowerShard: Free VRAM (text encoder / CLIP vision / VAE)",
    "PowerShardWanAnimate2Cache": "PowerShard Wan: Animate2 Cache (RAM workers)",
}
