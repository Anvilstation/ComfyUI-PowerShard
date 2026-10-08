"""LTX-2 / 2.3 / 2.5 (LTXAV, Lightricks): заголовок checkpoint, геометрия, опции. Без torch.

Геометрия повторяет comfy.model_detection (ветка ``adaln_single.emb.timestep_embedder``):
num_layers по transformer_blocks, attention_head_dim/cross_attention_dim по attn2.to_k,
переопределения из metadata ``config["transformer"]`` (2.3/2.5: cross_attention_adaln,
gated attention, rope_type, connectors ...), use_keyframes_abs_pos_embedding.
Полный checkpoint (VAE/аудио-VAE/vocoder/text projection в одном файле) допустим:
в workers читаются только ключи диффузионной модели.
"""
from dataclasses import dataclass, asdict
import json
import math
from .wan_config import WanCheckpoint, DTYPE_BYTES, KEY_PREFIXES, count_blocks, is_fp8_scale, quant_side_tensor

LTX_PROBE_KEY = "adaln_single.emb.timestep_embedder.linear_1.bias"


class LTXCheckpoint(WanCheckpoint):
    """Строгий reader заголовка LTX без чтения весов."""

    def __init__(self, path):
        super().__init__(path)

    def detect_prefix(self, header):
        for prefix in KEY_PREFIXES:
            if prefix + LTX_PROBE_KEY in header and prefix + "transformer_blocks.0.attn2.to_k.weight" in header:
                return prefix
        raise ValueError("Это не LTX-Video/LTX-2 diffusion checkpoint: нет adaln_single.emb.timestep_embedder "
                         "(поддерживаются префиксы model.diffusion_model., diffusion_model. и без префикса)")

    def model_config(self):
        return infer_ltx_config(self.tensors, self.metadata)

    def quantization(self):
        """LTX всегда деквантует (comfy_quant: nvfp4/mxfp8/int8/int4/fp8 -> fp16/bf16 по строкам); native int8 нет."""
        unknown = sorted(n for n, d in self.tensors.items() if n.endswith(".weight") and d["dtype"] in ("U8", "I8")
                         and "comfy_quant" not in d)
        if unknown:
            raise ValueError(f"LTX: упакованный вес без comfy_quant metadata ({unknown[0]} ...) — формат неизвестен")
        return {}


def infer_ltx_config(tensors, metadata=None):
    def shape(key):
        if key not in tensors:
            raise ValueError(f"LTX checkpoint неполный: нет {key}")
        return tensors[key]["shape"]
    config = {"image_model": "ltxav" if "audio_adaln_single.linear.weight" in tensors else "ltxv"}
    config["num_layers"] = count_blocks(tensors, "transformer_blocks.")
    k = shape("transformer_blocks.0.attn2.to_k.weight")
    config["attention_head_dim"] = k[0] // 32
    config["cross_attention_dim"] = k[1]
    if metadata and "config" in metadata:
        try:
            config.update(json.loads(metadata["config"]).get("transformer", {}))
        except (ValueError, AttributeError) as error:
            raise ValueError(f"LTX: metadata config не JSON: {error}") from None
    config["use_keyframes_abs_pos_embedding"] = "keyframes_abs_pos_embedding" in tensors
    infer_missing_ltx_keys(config, tensors)
    return config


def infer_missing_ltx_keys(config, tensors):
    """Файлы «только трансформер» иногда без metadata config: то, что однозначно видно по весам."""
    if config["image_model"] != "ltxav":
        return config
    config.setdefault("cross_attention_adaln", "transformer_blocks.0.prompt_scale_shift_table" in tensors)
    gate = tensors.get("transformer_blocks.0.attn1.to_gate_logits.weight")
    config.setdefault("apply_gated_attention", gate is not None)
    if gate is not None and "num_attention_heads" not in config:
        heads = gate["shape"][0]
        config["num_attention_heads"] = heads
        config["attention_head_dim"] = tensors["transformer_blocks.0.attn1.to_q.weight"]["shape"][0] // heads \
            if "transformer_blocks.0.attn1.to_q.weight" in tensors else config["attention_head_dim"]
    if "caption_proj_before_connector" not in config:
        if "caption_projection.linear_2.weight" in tensors:
            config["caption_proj_before_connector"] = False
        else:
            config["caption_proj_before_connector"] = True
            config.setdefault("caption_projection_first_linear", "caption_projection.linear_1.weight" in tensors)
    prefix = "video_embeddings_connector.transformer_1d_blocks."
    layers = {int(k[len(prefix):].split(".")[0]) for k in tensors if k.startswith(prefix)}
    if layers:
        config.setdefault("connector_num_layers", len(layers))
        cgate = tensors.get(prefix + "0.attn1.to_gate_logits.weight")
        config.setdefault("connector_apply_gated_attention", cgate is not None)
        width = tensors.get(prefix + "0.attn1.to_q.weight", {}).get("shape", [0])[0]
        if cgate is not None and width and "connector_num_attention_heads" not in config:
            config["connector_num_attention_heads"] = cgate["shape"][0]
            config["connector_attention_head_dim"] = width // cgate["shape"][0]
    return config


def ltx_geometry(config):
    """Числа, нужные host/probes: heads/head_dim видео и аудио."""
    heads = int(config.get("num_attention_heads", 32))
    head_dim = int(config["attention_head_dim"])
    av = config["image_model"] == "ltxav"
    return dict(av=av, heads=heads, head_dim=head_dim, inner_dim=heads * head_dim, num_layers=config["num_layers"],
                audio_heads=int(config.get("audio_num_attention_heads", 32)) if av else 0,
                audio_head_dim=int(config.get("audio_attention_head_dim", 64)) if av else 0,
                cross_attention_dim=config["cross_attention_dim"])


def describe_ltx(config):
    g = ltx_geometry(config)
    if not g["av"]:
        return f"ltxv_{g['num_layers']}L_{g['inner_dim']}"
    extra = []
    if config.get("cross_attention_adaln"):
        extra.append("xattn_adaln")
    if config.get("apply_gated_attention"):
        extra.append("gated")
    return f"ltxav_{g['num_layers']}L_{g['inner_dim']}" + ("_" + "_".join(extra) if extra else "")


def ltx_unet_config(config):
    value = dict(config)
    value["disable_unet_model_creation"] = True
    return value


def ltx_model_kwargs(config):
    return {k: v for k, v in config.items() if k != "disable_unet_model_creation"}


# Параметры, которые хранятся в FP32 (таблицы модуляции, маркеры, registers коннекторов).
FP32_PARAMETER_SUFFIXES = ("scale_shift_table", "keyframes_abs_pos_embedding", "learnable_registers")


def ltx_fp32_parameter(name):
    leaf = name.rsplit(".", 1)[-1]
    return any(token in leaf for token in FP32_PARAMETER_SUFFIXES)


def ltx_memory_plan(checkpoint, world_size=1):
    groups, target, stored = {}, 0, 0
    for name, d in checkpoint.tensors.items():
        n = math.prod(d["shape"])
        stored += math.prod(d.get("stored_shape", d["shape"])) * DTYPE_BYTES[d["dtype"]]
        if checkpoint.is_metadata(name) or is_fp8_scale(checkpoint, name) or quant_side_tensor(checkpoint, name):
            continue
        b = n * (4 if ltx_fp32_parameter(name) else 2)
        target += b
        parts = name.split(".")
        numeric = next((i for i, part in enumerate(parts) if part.isdigit()), None)
        group = ".".join(parts[:numeric + 1]) if numeric is not None else parts[0]
        groups[group] = groups.get(group, 0) + b
    return {"checkpoint_tensor_bytes": stored, "converted_storage_bytes": target, "world_size": world_size,
            "shard_bytes_lower_bound": math.ceil(target / world_size),
            "largest_group_bytes_upper_bound": max(groups.values(), default=0),
            "storage": checkpoint.storage()["kind"],
            "note_ru": "Оценка: + padding, buffers, active all-gather, NCCL/allocator, activations, attention."}


@dataclass(frozen=True)
class LTXOptions:
    """Числовая политика/память LTX и встроенные ускорители worker-а."""
    fp16_safe: bool = False
    debug_finite: bool = False
    mlp_chunk_mode: str = "off"
    mlp_chunk_tokens: int = 4096
    batch_chunk: int = 0
    attention_chunk: int = 8192   # ключей на шаг частичной v2a attention (online softmax)
    weight_dtype: str = "auto"    # auto: bf16 на sm80+ (RTX 30/40/50, A100/H100), fp16 на V100; либо fp16 / bf16
    weight_format: str = "dequantize"  # хранение Linear блоков: dequantize / as_file / int8 / fp8 / mxfp8 / nvfp4 / int4
    compute: str = "auto"              # auto / native / dequantize (для квантованного хранения)

    def __post_init__(self):
        if self.mlp_chunk_mode not in ("off", "auto", "manual"):
            raise ValueError("mlp_chunk_mode: off / auto / manual")
        if self.mlp_chunk_tokens < 1 or self.attention_chunk < 256:
            raise ValueError("mlp_chunk_tokens >= 1, attention_chunk >= 256")
        if self.batch_chunk < 0:
            raise ValueError("batch_chunk >= 0")
        if self.weight_dtype not in ("auto", "fp16", "bf16"):
            raise ValueError("weight_dtype: auto / fp16 / bf16")
        from .quant_formats import validate_quant_choice
        validate_quant_choice(self.weight_format, self.compute)

    def to_dict(self):
        return asdict(self)


def ltx_storage_ok(checkpoint, precision):
    kind = checkpoint.storage()["kind"]
    # precision (fp16 / int8_fp16) у LTX не задаёт хранение: любой формат деквантуется в weight_dtype.
    checkpoint.quantization()
    return kind
