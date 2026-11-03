"""Общие ноды Wan / LTX / MiniMax H3 / image-моделей: формат весов, LoRA, text encoder, checkpoint info.

Lazy imports: регистрация не импортирует torch/CUDA/ComfyUI.
"""
from dataclasses import dataclass, replace

from .quant_formats import WEIGHT_FORMATS, COMPUTE_MODES

CATEGORY = "PowerShard"


@dataclass(frozen=True)
class WeightSettings:
    """Хранение и вычисление Linear-весов в workers (один набор для всех loader-ов)."""
    weight_format: str = "dequantize"
    compute: str = "auto"
    weight_dtype: str = "auto"
    fp16_accumulation: bool = False


DEFAULT_WEIGHTS = WeightSettings()


def weights_or_default(weights):
    return weights if weights is not None else DEFAULT_WEIGHTS


def apply_weights(options, weights):
    """WanOptions / LTXOptions + WeightSettings -> новые options (dataclass replace, проверка в __post_init__)."""
    w = weights_or_default(weights)
    if w.weight_dtype == "fp32":
        raise ValueError("weight_dtype=fp32 — только для image-моделей; для Wan/LTX выберите auto / fp16 / bf16")
    return replace(options, weight_format=w.weight_format, compute=w.compute, weight_dtype=w.weight_dtype,
                   fp16_accumulation=bool(w.fp16_accumulation))


class PowerShardWeights:
    """Один узел вместо одинаковых опций в каждом loader-е. Не подключён — fp16/bf16 как в обычной ComfyUI."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "weight_format": (list(WEIGHT_FORMATS), {"default": "dequantize", "tooltip":
                "Как хранить Linear-веса в VRAM/RAM workers. dequantize — fp16/bf16. as_file — как в файле "
                "(int8/fp8/nvfp4/mxfp8/int4 остаются упакованными). int8/fp8/mxfp8/nvfp4/int4 — квантовать при загрузке. "
                "Экономит VRAM и трафик FSDP в 2–4 раза."}),
            "compute": (list(COMPUTE_MODES), {"default": "auto", "tooltip":
                "Как считать квантованные слои. native — низкобитные ядра comfy-kitchen (int8/int4: sm75+, fp8: sm89+, "
                "nvfp4/mxfp8: sm100+). dequantize — распаковка в fp16/bf16 на время GEMM (работает везде, в т.ч. V100). "
                "auto — native, если GPU и backend умеют формат."}),
            "weight_dtype": (["auto", "fp16", "bf16", "fp32"], {"default": "auto", "tooltip":
                "dtype весов и GEMM. auto — bf16 на sm80+ (RTX 30/40/50, A100/H100), fp16 на V100/T4; image-модели "
                "без fp16 (Z-Image, Qwen-Image, MOSS) на V100 — bf16 хранение + FP32 вычисление. fp32 — только image. "
                "MiniMax H3 всегда fp16; text encoder — FP32 вычисление."}),
            "fp16_accumulation": ("BOOLEAN", {"default": False, "tooltip":
                "fp16-накопление в cuBLAS GEMM генератора: до 2x на RTX 30/40/50 при fp16-весах; на V100/A100/H100 "
                "выигрыша нет. Для bf16-весов не действует."})}}
    RETURN_TYPES = ("POWERSHARD_WEIGHTS",)
    RETURN_NAMES = ("weights",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, weight_format="dequantize", compute="auto", weight_dtype="auto", fp16_accumulation=False):
        from .quant_formats import validate_quant_choice
        validate_quant_choice(weight_format, compute)
        return (WeightSettings(weight_format, compute, weight_dtype, bool(fp16_accumulation)),)


class PowerShardLoRA:
    """LoRA / LoHa / LoKr для любого loader-а PowerShard (Wan, LTX, MiniMax H3, image): сливается в shards."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"lora_name": (folder_paths.get_filename_list("loras"),),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "apply_to": (["all", "high_noise", "low_noise"], {"default": "all", "tooltip":
                                          "Только Wan 2.2 MoE A14B: к какому эксперту (Lightning/lightx2v — "
                                          "отдельные файлы high/low). Остальные модели — all."})},
                "optional": {"lora": ("POWERSHARD_LORA",)}}
    RETURN_TYPES = ("POWERSHARD_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(self, lora_name, strength=1.0, apply_to="all", lora=None):
        import folder_paths
        from .wan_config import WanLoraSpec
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return (tuple(lora or ()) + (WanLoraSpec(str(path), float(strength), apply_to),),)

    @classmethod
    def IS_CHANGED(cls, lora_name, strength=1.0, apply_to="all", lora=None):
        from .wan_nodes import _stat
        return (_stat("loras", lora_name), strength, apply_to, repr(lora))


# type ноды -> CLIPType ComfyUI (кроме wan_umt5 / ltx_gemma со своими путями загрузки)
TEXT_ENCODERS = {"wan_umt5": "wan", "ltx_gemma": "ltxv", "flux": "flux", "flux2": "flux2", "flux3": "flux3",
                 "chroma": "chroma", "qwen_image": "qwen_image", "z_image": "lumina2", "lumina2": "lumina2",
                 "moss_image": "moss_image", "fibo2": "fibo2", "hidream": "hidream", "sd3": "sd3",
                 "sdxl": "stable_diffusion", "sd15": "stable_diffusion", "hunyuan_image": "hunyuan_image",
                 "krea2": "krea2"}


class PowerShardTextEncoder:
    """Text encoder Wan / LTX / image-моделей (T5, Qwen, Mistral, Gemma, CLIP) в пуле workers или на CPU FP32."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        from .ltx_nodes import _model_choices
        return {"required": {
            "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
            "type": (list(TEXT_ENCODERS), {"default": "wan_umt5", "tooltip":
                "wan_umt5 — umT5-XXL для Wan 2.1/2.2. ltx_gemma — Gemma для LTX-2/2.3/2.5 (нужен ltx_checkpoint). "
                "flux — clip_l + t5xxl (второй файл в text_encoder_2); flux2 — Mistral/Qwen3; qwen_image — "
                "Qwen2.5-VL; z_image — Qwen3-4B; moss_image — Qwen3.5-9B; sdxl — clip_l + clip_g; sd3 / hidream — "
                "до 3–4 файлов. Тип как в родном CLIPLoader / DualCLIPLoader."}),
            "placement": (["workers", "cpu_fp32"], {"default": "workers", "tooltip":
                "workers — веса шардируются по GPU из config (FSDP, FP32 вычисление), host GPU не занимается. "
                "cpu_fp32 — родной энкодер ComfyUI на CPU в FP32 (медленнее, ~23 GB RAM для umT5, без config)."}),
            "after_encode": (["release", "release_now", "keep_ram"], {"default": "release", "tooltip":
                "workers: release — закрыть при старте генератора / Free VRAM; release_now — сразу после encode; "
                "keep_ram — держать workers между генерациями: при старте генератора веса паркуются в RAM и "
                "возвращаются на GPU без перечитывания файла (секунды вместо загрузки). Одинаковый текст берётся из кэша."})},
            "optional": {
                "config": ("POWERSHARD_CONFIG", {"tooltip": "Тот же config, что у генератора (обязателен для workers)"}),
                "text_encoder_2": (["none"] + folder_paths.get_filename_list("text_encoders"), {"default": "none",
                                   "tooltip": "Второй файл энкодера (flux: t5xxl к clip_l; sdxl: clip_g; sd3/hidream)"}),
                "text_encoder_3": (["none"] + folder_paths.get_filename_list("text_encoders"), {"default": "none"}),
                "text_encoder_4": (["none"] + folder_paths.get_filename_list("text_encoders"), {"default": "none"}),
                "ltx_checkpoint": (["none"] + _model_choices(), {"default": "none", "tooltip": "Только ltx_gemma: LTX checkpoint с "
                                                               "text_embedding_projection (как в LTXAVTextEncoderLoader)"}),
                "weights": ("POWERSHARD_WEIGHTS", {"tooltip": "workers (кроме wan_umt5): хранение весов энкодера"})}}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @staticmethod
    def _paths(text_encoder, type, ltx_checkpoint, extra=()):
        import folder_paths
        from .ltx_nodes import _resolve
        paths = [str(folder_paths.get_full_path_or_raise("text_encoders", text_encoder))]
        paths += [str(folder_paths.get_full_path_or_raise("text_encoders", name)) for name in extra
                  if name and name != "none"]
        if type == "ltx_gemma":
            if not ltx_checkpoint or ltx_checkpoint == "none":
                raise ValueError("ltx_gemma: выберите ltx_checkpoint (файл LTX с text_embedding_projection)")
            paths.append(str(_resolve(ltx_checkpoint)))
        return paths

    def load(self, text_encoder, type="wan_umt5", placement="workers", after_encode="release", config=None,
             text_encoder_2="none", text_encoder_3="none", text_encoder_4="none", ltx_checkpoint="none", weights=None):
        import folder_paths
        from .wan_nodes import _report_dir
        if type not in TEXT_ENCODERS:
            raise ValueError(f"type: {' / '.join(TEXT_ENCODERS)}")
        extra = (text_encoder_2, text_encoder_3, text_encoder_4)
        if type == "wan_umt5" and any(x not in (None, "none") for x in extra):
            raise ValueError("wan_umt5: один файл umT5-XXL, text_encoder_2..4 оставьте none")
        paths = self._paths(text_encoder, type, ltx_checkpoint, extra)
        embeddings = folder_paths.get_folder_paths("embeddings")
        if placement == "cpu_fp32":
            import torch
            import comfy.sd
            clip_type = getattr(comfy.sd.CLIPType, TEXT_ENCODERS[type].upper())
            options = {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu"), "dtype": torch.float32}
            return (comfy.sd.load_clip(paths, embedding_directory=embeddings, clip_type=clip_type, model_options=options),)
        if config is None:
            raise ValueError("placement=workers: подключите POWERSHARD_CONFIG (тот же, что у генератора)")
        if type == "wan_umt5":
            from .wan_text import load_wan_t5
            return (load_wan_t5(paths[0], config, after_encode, _report_dir(), embeddings),)
        from .native_te import load_distributed_te
        w = weights_or_default(weights)
        return (load_distributed_te(paths, TEXT_ENCODERS[type], config, after_encode, _report_dir(), embeddings,
                                    w.weight_format, w.compute),)

    @classmethod
    def IS_CHANGED(cls, text_encoder, type="wan_umt5", placement="workers", after_encode="release", config=None,
                   text_encoder_2="none", text_encoder_3="none", text_encoder_4="none", ltx_checkpoint="none",
                   weights=None):
        from .wan_nodes import _stat
        from .ltx_nodes import _choice_stat
        ltx = _choice_stat(ltx_checkpoint) if type == "ltx_gemma" and ltx_checkpoint not in (None, "none") else None
        extra = [_stat("text_encoders", x) for x in (text_encoder_2, text_encoder_3, text_encoder_4)
                 if x not in (None, "none")]
        return (_stat("text_encoders", text_encoder), type, placement, after_encode, repr(config), ltx, extra,
                repr(weights))


def image_info(path, gpu_count):
    """Image-модель: семейство по ComfyUI, число параметров, dtypes, sequence parallel (без чтения весов)."""
    from .image_backend import build_native
    from .image_parallel import ParallelContext, install
    import torch
    model, model_config, _, _, header, prefix = build_native(str(path), torch.float16, torch.float16)
    network = model.diffusion_model
    params = sum(p.numel() for p in network.parameters())
    ctx = ParallelContext(max(1, gpu_count), 0)
    families = install(network, ctx)
    dtypes = [str(d).replace("torch.", "") for d in model_config.supported_inference_dtypes]
    return dict(model="image", family=type(model).__name__, diffusion_model=type(network).__name__,
                parameters=params, key_prefix=prefix, supported_dtypes=dtypes, sequence_parallel=families or None,
                memory=dict(fp16_total_bytes=params * 2, fp16_shard_bytes=-(-params * 2 // max(1, gpu_count))))


class PowerShardInfo:
    """Геометрия, формат и оценка памяти checkpoint Wan / LTX / image без загрузки весов."""
    @classmethod
    def INPUT_TYPES(cls):
        from .ltx_nodes import _model_choices
        return {"required": {"checkpoint": (_model_choices(),),
                             "gpu_count": ("INT", {"default": 6, "min": 1, "max": 64})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def inspect(self, checkpoint, gpu_count=6):
        import json
        from .ltx_nodes import _resolve
        path = _resolve(checkpoint)
        errors = []
        try:
            from .wan_config import WanCheckpoint, describe_family, memory_plan
            ckpt = WanCheckpoint(path)
            geometry = ckpt.model_config()
            data = dict(model="wan", family=describe_family(geometry), geometry=geometry, storage=ckpt.storage(),
                        key_prefix=ckpt.prefix, memory=memory_plan(ckpt, gpu_count))
        except Exception as error:
            errors.append(f"Wan: {error}")
            try:
                from .ltx_config import LTXCheckpoint, describe_ltx, ltx_geometry, ltx_memory_plan
                ckpt = LTXCheckpoint(path)
                config = ckpt.model_config()
                data = dict(model="ltx", family=describe_ltx(config), config=config, geometry=ltx_geometry(config),
                            key_prefix=ckpt.prefix, ignored_non_dit_keys=len(ckpt.ignored),
                            memory=ltx_memory_plan(ckpt, gpu_count))
            except Exception as error2:
                errors.append(f"LTX: {error2}")
                try:
                    data = image_info(path, gpu_count)
                except Exception as error3:
                    errors.append(f"image: {error3}")
                    data = dict(model="unknown", errors=errors)
        value = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return {"ui": {"text": [value]}, "result": (value,)}


from .image_nodes import IMAGE_NODE_CLASS_MAPPINGS, IMAGE_NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

COMMON_NODE_CLASS_MAPPINGS = {
    "PowerShardWeights": PowerShardWeights,
    "PowerShardLoRA": PowerShardLoRA,
    "PowerShardTextEncoder": PowerShardTextEncoder,
    "PowerShardInfo": PowerShardInfo,
    **IMAGE_NODE_CLASS_MAPPINGS,
}
COMMON_NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerShardWeights": "PowerShard: формат весов / вычисление",
    "PowerShardLoRA": "PowerShard: LoRA (Wan / LTX / H3 / image)",
    "PowerShardTextEncoder": "PowerShard: Text Encoder (Wan / LTX / Flux / Qwen / Z-Image / MOSS ...)",
    "PowerShardInfo": "PowerShard: checkpoint info (Wan / LTX / image)",
    **IMAGE_NODE_DISPLAY_NAME_MAPPINGS,
}
