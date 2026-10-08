"""Wan 2.1/2.2: заголовки safetensors, геометрия, эксперты MoE и LoRA specs.

Модуль не импортирует torch: его используют ноды до запуска workers и тесты
без CUDA. Ничего из существующего H3 кода не меняется; формат fp8_scaled и
префиксы ключей Wan здесь разбираются отдельно от ``checkpoint.Checkpoint``.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import struct

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2,
               "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
FP8_DTYPES = ("F8_E4M3", "F8_E5M2")
FLOAT_DTYPES = ("F16", "BF16", "F32", "F64") + FP8_DTYPES
KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "")
# Ключи, которые относятся к native ComfyUI quant/attention metadata, а не к весам.
METADATA_SUFFIXES = (".comfy_quant", ".scale_input", ".input_scale", ".comfy_attention.config")
FP8_SCALE_SUFFIXES = (".scale_weight", ".weight_scale")
METADATA_KEYS = ("scaled_fp8",)
# Варианты, которые распознаются, но не исполняются PowerShard (пусто: все варианты comfy Wan поддержаны).
UNSUPPORTED_VARIANTS = {}
# model_type, которые нельзя отличить по весам (checkpoint i2v-формы): задаются metadata или опцией loader.
MODEL_TYPE_OVERRIDES = ("auto", "animate2")
# Официальные значения Wan 2.2 (configs/wan_t2v_A14B.py, wan_i2v_A14B.py):
# переключение экспертов по timestep = boundary * 1000.
MOE_BOUNDARY = {"t2v": 0.875, "i2v": 0.900}
EXPERT_SLOTS = ("high", "low", "main")


def read_safetensors_header(path):
    """Строгий разбор заголовка: dtype/shape/смещения без пропусков и перекрытий."""
    path = Path(path).expanduser().resolve(strict=True)
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("Обрезанный safetensors")
        header_size = struct.unpack("<Q", raw)[0]
        if not 2 <= header_size <= 64 * 1024 * 1024:
            raise ValueError("Недопустимый размер заголовка safetensors")
        raw = f.read(header_size)
    header = json.loads(raw)
    metadata = header.pop("__metadata__", {}) or {}
    data_start = 8 + header_size
    size = path.stat().st_size
    intervals = []
    for name, desc in header.items():
        shape, dtype = desc["shape"], desc["dtype"]
        a, b = desc["data_offsets"]
        if dtype not in DTYPE_BYTES or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Неподдерживаемый dtype/shape {dtype}: {name}")
        if a < 0 or b - a != math.prod(shape) * DTYPE_BYTES[dtype] or data_start + b > size:
            raise ValueError(f"Некорректный диапазон tensor: {name}")
        intervals.append((a, b))
    end = 0
    for a, b in sorted(intervals):
        if a != end:
            raise ValueError("Перекрывающиеся tensors или разрыв safetensors")
        end = b
    if data_start + end != size:
        raise ValueError("Размер файла не соответствует заголовку")
    return path, hashlib.sha256(raw).hexdigest(), metadata, header, data_start


class WanCheckpoint:
    """Строгий reader заголовка safetensors для Wan без чтения весов."""

    def __init__(self, path, model_type="auto"):
        self.path, self.header_hash, self.metadata, header, self.data_start = read_safetensors_header(path)
        if model_type not in MODEL_TYPE_OVERRIDES:
            raise ValueError(f"model_type override: {MODEL_TYPE_OVERRIDES}")
        self.model_type = model_type
        self.raw = header
        self.prefix = self.detect_prefix(header)
        # tensors: имя модели -> descriptor + исходный ключ файла (+ row_offset для fused tensors).
        self.tensors = {}
        self.ignored = []
        for key, desc in header.items():
            name = self.rename(key)
            if name is None:
                self.ignored.append(key)
                continue
            self.tensors[name] = dict(desc, key=key)
        split_fused_in_proj(self.tensors)
        # comfy_quant (nvfp4/mxfp8/int8/int4/...): логические формы весов; деквантование — в worker.
        from .quant_formats import annotate_quantized
        self.quant_configs = annotate_quantized(self.tensors, self.read_json_tensor)

    def detect_prefix(self, header):
        return detect_prefix(header)

    def rename(self, key):
        return key[len(self.prefix):] if key.startswith(self.prefix) else None

    def identity(self):
        st = self.path.stat()
        return {"path": str(self.path), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                "header_sha256": self.header_hash, "key_prefix": self.prefix}

    def stamp(self):
        st = self.path.stat()
        return [str(self.path), st.st_size, st.st_mtime_ns]

    def read_json_tensor(self, name):
        desc = self.tensors[name]
        a, b = desc["data_offsets"]
        if desc["dtype"] != "U8" or b - a > 1 << 20:
            raise ValueError(f"Неверные quant metadata: {name}")
        with self.path.open("rb") as f:
            f.seek(self.data_start + a)
            return json.loads(f.read(b - a))

    def model_config(self):
        return infer_wan_config(self.tensors, self.metadata, self.model_type)

    def is_metadata(self, name):
        return name in METADATA_KEYS or name.endswith(METADATA_SUFFIXES)

    def fp8_scale_key(self, weight_name):
        """Ключ масштаба fp8_scaled для ``X.weight`` либо None (unscaled fp8)."""
        if not weight_name.endswith(".weight"):
            return None
        module = weight_name[:-len(".weight")]
        for suffix in FP8_SCALE_SUFFIXES:
            key = module + suffix
            if key in self.tensors and self.tensors[key]["dtype"] in ("F32", "F16", "BF16"):
                return key
        return None

    def quantization(self):
        """Native int8_tensorwise хранение (precision=int8_fp16, ядра H3). Остальные форматы деквантуются
        загрузчиком (quant_formats) и этим методом не требуются."""
        other = sorted({c["format"] for c in self.quant_configs.values()} - {"int8_tensorwise"})
        if other:
            raise ValueError(f"precision=int8_fp16 хранит только int8_tensorwise; в checkpoint также {other}. "
                             "Выберите precision=fp16 — веса будут деквантованы при загрузке")
        configs = {}
        for name, d in self.tensors.items():
            if d["dtype"] != "I8":
                continue
            if name.endswith(".weight") and d.get("comfy_quant", {}).get("format") not in (None, "int8_tensorwise"):
                continue
            if not name.endswith(".weight") or len(d["shape"]) != 2:
                raise ValueError(f"INT8 вне Linear: {name}")
            module = name[:-7]
            if module + ".comfy_quant" not in self.tensors:
                raise ValueError(f"INT8 без comfy_quant metadata: {module}")
            conf = self.read_json_tensor(module + ".comfy_quant")
            if conf.get("format") != "int8_tensorwise":
                raise ValueError(f"Неподдерживаемое квантование {module}: {conf}")
            params = conf.get("params", {})
            gs = int(conf.get("convrot_groupsize", params.get("convrot_groupsize", 256)))
            convrot = bool(conf.get("convrot", params.get("convrot", False)))
            if convrot and (gs < 4 or gs & (gs - 1) or (gs.bit_length() - 1) % 2 or d["shape"][1] % gs):
                raise ValueError(f"Некорректная ConvRot-группа: {module}")
            scale = self.tensors.get(module + ".weight_scale")
            if scale is None or scale["shape"] != [d["shape"][0], 1] or scale["dtype"] != "F32":
                raise ValueError(f"Ожидался row-wise FP32 scale: {module}")
            configs[module] = {"convrot": convrot, "group_size": gs}
        return configs

    def storage(self):
        """Краткое описание формата для отчёта и проверки precision."""
        counts = {}
        for name, d in self.tensors.items():
            if self.is_metadata(name):
                continue
            counts[d["dtype"]] = counts.get(d["dtype"], 0) + math.prod(d.get("stored_shape", d["shape"]))
        formats = sorted({c["format"] for c in self.quant_configs.values()})
        if formats and formats != ["int8_tensorwise"] and not set(formats) <= {"float8_e4m3fn", "float8_e5m2"}:
            kind = "comfy_quant:" + "+".join(formats)
        elif counts.get("I8"):
            kind = "int8_tensorwise"
        elif any(counts.get(x) for x in FP8_DTYPES):
            scaled = any(self.fp8_scale_key(n) for n, d in self.tensors.items() if d["dtype"] in FP8_DTYPES)
            kind = "fp8_scaled" if scaled else "fp8"
        elif counts.get("BF16", 0) >= counts.get("F16", 0) and counts.get("BF16"):
            kind = "bf16"
        elif counts.get("F16"):
            kind = "fp16"
        else:
            kind = "fp32"
        return dict(kind=kind, elements_by_dtype=counts)


def detect_prefix(header):
    for prefix in KEY_PREFIXES:
        if prefix + "head.modulation" in header:
            return prefix
    raise ValueError("Это не Wan 2.1/2.2 diffusion checkpoint: нет head.modulation "
                     "(поддерживаются префиксы model.diffusion_model., diffusion_model. и без префикса)")


def split_fused_in_proj(tensors):
    """WanDancer music_encoder: nn.MultiheadAttention in_proj [3d, ...] -> q/k/v_proj (как comfy
    supported_models.WAN22_WanDancer.process_unet_state_dict), без копирования: смещение строк."""
    for name in [n for n in tensors if "music_encoder" in n and ".self_attn.in_proj_" in n]:
        desc = tensors.pop(name)
        suffix = "weight" if name.endswith("weight") else "bias"
        prefix = name[:-len("in_proj_" + suffix)]
        rows = desc["shape"][0]
        if rows % 3:
            raise ValueError(f"WanDancer: {name} не делится на q/k/v")
        d = rows // 3
        for i, part in enumerate(("q_proj", "k_proj", "v_proj")):
            tensors[f"{prefix}{part}.{suffix}"] = dict(desc, shape=[d] + list(desc["shape"][1:]),
                                                      row_offset=desc.get("row_offset", 0) + i * d)


def count_blocks(tensors, prefix="blocks."):
    ids = {int(k[len(prefix):].split(".")[0]) for k in tensors if k.startswith(prefix)}
    if ids != set(range(len(ids))):
        raise ValueError(f"Непоследовательные блоки {prefix}")
    return len(ids)


def infer_wan_config(tensors, metadata=None, model_type="auto"):
    """Повторяет comfy.model_detection для Wan (VACE/Camera/S2V/HuMo/Animate/SCAIL/SCAIL2/WanDancer,
    Animate2 через metadata/override), строго и без весов."""
    for key, label in UNSUPPORTED_VARIANTS.items():
        if key in tensors:
            raise ValueError(f"{label} checkpoint не поддерживается PowerShard Wan")

    def shape(key):
        if key not in tensors:
            raise ValueError(f"Wan checkpoint неполный: нет {key}")
        return tensors[key]["shape"]

    dim = shape("head.modulation")[-1]
    head_dim = 128
    if dim % head_dim:
        raise ValueError(f"Wan dim={dim} не кратен head_dim=128")
    has_img = "img_emb.proj.0.bias" in tensors
    config = dict(
        image_model="wan2.1",
        model_type="i2v" if has_img else "t2v",
        patch_size=(1, 2, 2), text_len=512,
        in_dim=shape("patch_embedding.weight")[1], dim=dim,
        ffn_dim=shape("blocks.0.ffn.0.weight")[0],
        freq_dim=shape("time_embedding.0.weight")[1],
        text_dim=shape("text_embedding.0.weight")[1],
        out_dim=shape("head.head.weight")[0] // 4,
        num_heads=dim // head_dim, num_layers=count_blocks(tensors),
        window_size=(-1, -1),
        qk_norm="blocks.0.self_attn.norm_q.weight" in tensors,
        cross_attn_norm="blocks.0.norm3.weight" in tensors,
        eps=1e-6)
    if not config["qk_norm"]:
        raise ValueError("Wan без q/k RMSNorm не поддерживается FP16 attention bounds")
    if tuple(shape("patch_embedding.weight")[2:]) != (1, 2, 2):
        raise ValueError("Ожидался patch_size (1,2,2)")
    if "vace_patch_embedding.weight" in tensors:
        config.update(model_type="vace", vace_in_dim=shape("vace_patch_embedding.weight")[1],
                      vace_layers=count_blocks(tensors, "vace_blocks."))
        if has_img:
            config["vace_image_input"] = True
    elif "control_adapter.conv.weight" in tensors:
        config["model_type"] = "camera" if has_img else "camera_2.2"
        config["in_dim_control_adapter"] = shape("control_adapter.conv.weight")[1] // 64
    elif "casual_audio_encoder.encoder.final_linear.weight" in tensors:
        config["model_type"] = "s2v"
        injectors = count_blocks(tensors, "audio_injector.injector.")
        if injectors != 12:
            raise ValueError(f"Wan S2V: {injectors} аудио-инъекторов, comfy WanModel_S2V ожидает 12 (слои 0,4,...,39)")
    elif "audio_proj.audio_proj_glob_1.layer.bias" in tensors:
        config["model_type"] = "humo"
        tokens = shape("audio_proj.audio_proj_glob_3.layer.weight")[0] // shape("audio_proj.audio_proj_glob_norm.layer.weight")[0]
        if tokens != 16:
            raise ValueError(f"Wan HuMo: {tokens} аудио-токенов на кадр, comfy WanT2VCrossAttentionGather ожидает 16")
    elif "face_adapter.fuser_blocks.0.k_norm.weight" in tensors:
        config["model_type"] = "animate"
    elif "patch_embedding_mask.weight" in tensors:
        config["model_type"] = "scail2"
        config["mask_in_dim"] = shape("patch_embedding_mask.weight")[1]
    elif "patch_embedding_pose.weight" in tensors:
        config["model_type"] = "scail"
    elif "patch_embedding_global.weight" in tensors:
        config["model_type"] = "wandancer"
        injectors = count_blocks(tensors, "music_injector.injector.")
        if injectors != 8:
            raise ValueError(f"WanDancer: {injectors} music-инъекторов, comfy WanDancerModel ожидает 8 (слои 0,4,...,27)")
        config["music_feature_dim"] = shape("music_projection.weight")[1]
        config["music_latent_dim"] = shape("music_projection.weight")[0]
        config["music_dim"] = shape("music_encoder.0.norm1.weight")[0]
    if "img_emb.emb_pos" in tensors:
        config["flf_pos_embed_token_number"] = shape("img_emb.emb_pos")[1]
    if "ref_conv.weight" in tensors:
        config["in_dim_ref_conv"] = shape("ref_conv.weight")[1]
    if metadata and "config" in metadata:
        try:
            override = json.loads(metadata["config"]).get("transformer", {})
        except (ValueError, AttributeError):
            override = {}
        for key in ("eps", "freq_dim", "text_len"):
            if key in override:
                config[key] = override[key]
        if override.get("model_type") == "animate2":  # comfy: dit_config.update(metadata transformer)
            model_type = "animate2"
    if model_type == "animate2":
        if config["model_type"] != "i2v" or config["in_dim"] != 36:
            raise ValueError(f"Wan Animate2 ожидает checkpoint формы Wan2.1 I2V (in_dim 36 + img_emb), "
                             f"а этот распознан как {config['model_type']} (in_dim {config['in_dim']})")
        config["model_type"] = "animate2"
    return config


def describe_family(config):
    kind = config["model_type"]
    if kind == "vace":
        return "wan_vace" + ("_i2v" if config.get("vace_image_input") else "")
    if kind == "s2v":
        return "wan2.2_s2v"
    if kind == "animate":
        return "wan2.2_animate"
    if kind in ("humo", "scail", "scail2", "wandancer", "animate2"):
        return {"humo": "wan_humo", "scail": "wan_scail", "scail2": "wan_scail2", "wandancer": "wan2.2_wandancer",
                "animate2": "wan_animate2"}[kind]
    if kind.startswith("camera"):
        return "wan_fun_" + kind
    if config["out_dim"] == 48:
        return "wan2.2_ti2v_5b"
    if config["model_type"] == "i2v":
        return "wan2.1_i2v_or_flf" if config.get("flf_pos_embed_token_number") is None else "wan2.1_flf2v"
    if config["in_dim"] == 36:
        return "wan2.2_i2v_a14b_expert"
    return "wan_t2v"


def comfy_unet_config(config):
    """unet_config для comfy.model_detection.model_config_from_unet_config."""
    value = {k: v for k, v in config.items() if k not in ("text_len",)}
    value["disable_unet_model_creation"] = True
    return value


def model_kwargs(config):
    """Аргументы comfy.ldm.wan.model.WanModel (без служебных ключей host)."""
    return {k: v for k, v in config.items() if k not in ("disable_unet_model_creation",)}


def same_geometry(a, b):
    keys = ("dim", "num_heads", "num_layers", "ffn_dim", "in_dim", "out_dim", "text_dim", "model_type", "freq_dim")
    return all(a.get(k) == b.get(k) for k in keys)


def is_fp8_scale(checkpoint, name):
    for suffix in FP8_SCALE_SUFFIXES:
        if name.endswith(suffix):
            weight = checkpoint.tensors.get(name[:-len(suffix)] + ".weight")
            return weight is not None and weight["dtype"] in FP8_DTYPES
    return False


def quant_side_tensor(checkpoint, name):
    """Scale/codebook/... тензоры comfy_quant модулей (кроме weight/bias)."""
    module, _, leaf = name.rpartition(".")
    return module in getattr(checkpoint, "quant_configs", {}) and leaf not in ("weight", "bias")


def fp32_parameter(name):
    """Параметры, которые PowerShard Wan хранит в FP32 (как comfy patch_embedding)."""
    return name.endswith("modulation") or name.startswith("patch_embedding")


def memory_plan(checkpoint, world_size=1):
    """Оценка постоянных байтов после конвертации: fp8/bf16 -> fp16, fp32 islands."""
    groups, target, stored = {}, 0, 0
    for name, d in checkpoint.tensors.items():
        n = math.prod(d["shape"])
        stored += math.prod(d.get("stored_shape", d["shape"])) * DTYPE_BYTES[d["dtype"]]
        if checkpoint.is_metadata(name) or is_fp8_scale(checkpoint, name) or quant_side_tensor(checkpoint, name):
            continue  # metadata / scales: уже умножены в FP16/BF16 веса
        native_int8 = d["dtype"] == "I8" and d.get("comfy_quant", {}).get("format") in (None, "int8_tensorwise")
        b = n * (1 if native_int8 else 4 if fp32_parameter(name) or name.endswith(".weight_scale") else 2)
        target += b
        parts = name.split(".")
        numeric = next((i for i, part in enumerate(parts) if part.isdigit()), None)
        group = ".".join(parts[:numeric + 1]) if numeric is not None else parts[0]
        groups[group] = groups.get(group, 0) + b
    return {"checkpoint_tensor_bytes": stored, "converted_storage_bytes": target, "world_size": world_size,
            "shard_bytes_lower_bound": math.ceil(target / world_size),
            "largest_group_bytes_upper_bound": max(groups.values(), default=0),
            "note_ru": "Оценка: + padding, buffers, active all-gather, NCCL/allocator, activations, attention. Не гарантия размещения."}


@dataclass(frozen=True)
class WanOptions:
    """Числовая политика/память Wan; общая для всех экспертов одной session."""
    fp16_safe: bool = False
    debug_finite: bool = False
    mlp_chunk_mode: str = "off"
    mlp_chunk_tokens: int = 4096
    moe_residency: str = "swap"
    batch_chunk: int = 0  # 0: весь CFG batch одним forward; 1: cond/uncond последовательно в worker
    model_type: str = "auto"  # auto / animate2 (Animate2 checkpoint имеет форму Wan2.1 I2V)
    weight_dtype: str = "auto"  # auto: bf16 на sm80+ (RTX 30/40/50, A/H100), fp16 на V100/T4; либо fp16 / bf16
    weight_format: str = "dequantize"  # хранение Linear блоков: dequantize / as_file / int8 / fp8 / mxfp8 / nvfp4 / int4
    compute: str = "auto"              # auto / native / dequantize (для квантованного хранения)

    def __post_init__(self):
        if self.mlp_chunk_mode not in ("off", "auto", "manual"):
            raise ValueError("mlp_chunk_mode: off / auto / manual")
        if self.mlp_chunk_tokens < 1:
            raise ValueError("mlp_chunk_tokens должен быть положительным")
        if self.moe_residency not in ("both", "swap"):
            raise ValueError("moe_residency: both / swap")
        if self.batch_chunk < 0:
            raise ValueError("batch_chunk >= 0")
        if self.model_type not in MODEL_TYPE_OVERRIDES:
            raise ValueError(f"model_type: {' / '.join(MODEL_TYPE_OVERRIDES)}")
        if self.weight_dtype not in ("auto", "fp16", "bf16"):
            raise ValueError("weight_dtype: auto / fp16 / bf16")
        from .quant_formats import validate_quant_choice
        validate_quant_choice(self.weight_format, self.compute)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class WanLoraSpec:
    path: str
    strength: float = 1.0
    apply_to: str = "all"  # all / high_noise / low_noise

    def __post_init__(self):
        if self.apply_to not in ("all", "high_noise", "low_noise"):
            raise ValueError("apply_to: all / high_noise / low_noise")
        if not math.isfinite(self.strength):
            raise ValueError("LoRA strength должен быть конечным")

    def targets(self, slot):
        if self.apply_to == "all":
            return True
        if slot == "main":
            raise ValueError(f"LoRA {Path(self.path).name}: apply_to={self.apply_to} требует MoE loader (high/low)")
        return (self.apply_to == "high_noise") == (slot == "high")

    def stamp(self):
        st = Path(self.path).stat()
        return dict(path=str(Path(self.path).resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns,
                    strength=float(self.strength), apply_to=self.apply_to)


UNI3C_RENAMES = ((".self_attn.to_q.", ".self_attn.q."), (".self_attn.to_k.", ".self_attn.k."),
                 (".self_attn.to_v.", ".self_attn.v."), (".self_attn.to_out.0.", ".self_attn.o."))


class Uni3CCheckpoint(WanCheckpoint):
    """Uni3C ControlNet (comfy model_patches): diffusers-имена attention -> имена WanSelfAttention."""

    def detect_prefix(self, header):
        if "controlnet_patch_embedding.weight" not in header:
            raise ValueError("Это не Uni3C ControlNet: нет controlnet_patch_embedding.weight")
        return ""

    def rename(self, key):
        for old, new in UNI3C_RENAMES:
            key = key.replace(old, new)
        return key

    def model_config(self):
        t = self.tensors

        def shape(key):
            if key not in t:
                raise ValueError(f"Uni3C checkpoint неполный: нет {key}")
            return t[key]["shape"]
        conv_out = shape("controlnet_patch_embedding.weight")[0]
        mask = shape("controlnet_mask_embedding.mask_proj.0.weight")
        return dict(in_channels=shape("controlnet_patch_embedding.weight")[1], conv_out_dim=conv_out,
                    dim=shape("proj_in.weight")[0] if "proj_in.weight" in t else conv_out,
                    ffn_dim=shape("controlnet_blocks.0.ffn.0.bias")[0],
                    num_layers=sum(1 for k in t if k.startswith("proj_out.") and k.endswith(".weight")),
                    time_embed_dim=shape("controlnet_blocks.0.norm1.linear.weight")[1],
                    out_proj_dim=shape("proj_out.0.weight")[0], add_channels=mask[1], mid_channels=mask[0])


class MultiTalkCheckpoint(WanCheckpoint):
    """InfiniteTalk / MultiTalk model patch (comfy model_patches): audio_proj (host) + 40 audio cross-attn блоков.

    audio_proj маленький и считается на host (native project_audio_features), в workers
    грузятся только blocks.* — Shard(0) как у генератора.
    """

    def detect_prefix(self, header):
        if "audio_proj.proj1.weight" not in header or "blocks.0.audio_cross_attn.proj.weight" not in header:
            raise ValueError("Это не InfiniteTalk/MultiTalk model patch: нет audio_proj.proj1 / blocks.0.audio_cross_attn")
        return ""

    def rename(self, key):
        return None if key.startswith("audio_proj.") else key

    def model_config(self):
        t = self.tensors
        return dict(in_dim=t["blocks.0.audio_cross_attn.proj.weight"]["shape"][0],
                    out_dim=t["blocks.0.audio_cross_attn.kv_linear.weight"]["shape"][1],
                    num_layers=count_blocks(t))


T5_PREFIXES = ("", "umt5xxl.transformer.", "text_encoders.umt5xxl.transformer.", "t5xxl.transformer.")
T5_ROOT = "umt5xxl.transformer."


class T5Checkpoint(WanCheckpoint):
    """umT5-XXL safetensors (fp16/bf16/fp8_scaled): имена файла -> имена comfy WanT5Model."""

    def detect_prefix(self, header):
        for prefix in T5_PREFIXES:
            if prefix + "encoder.final_layer_norm.weight" in header and prefix + "shared.weight" in header:
                return prefix
        raise ValueError("Это не umT5/T5 encoder checkpoint: нет encoder.final_layer_norm.weight / shared.weight")

    def rename(self, key):
        if key == "spiece_model" or not key.startswith(self.prefix):
            return None
        return T5_ROOT + key[len(self.prefix):]

    def model_config(self):
        shared = self.tensors[T5_ROOT + "shared.weight"]["shape"]
        return dict(vocab_size=shared[0], d_model=shared[1],
                    num_layers=count_blocks({k[len(T5_ROOT):]: v for k, v in self.tensors.items()}, "encoder.block."))


def lora_plan_for(slot, loras):
    """JSON-описание LoRA, которые worker сливает в локальные строки эксперта."""
    return [l.stamp() for l in loras if l.strength != 0 and l.targets(slot)]
