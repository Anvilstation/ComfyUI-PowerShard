"""Любые форматы весов ComfyUI в FSDP-загрузчиках PowerShard (Wan, LTX, umT5, Gemma, ControlNet/patches).

Форматы с метаданными ``<module>.comfy_quant`` (как их сохраняет ComfyUI / comfy-kitchen):
float8_e4m3fn / float8_e5m2 (+weight_scale), mxfp8, nvfp4, int8_tensorwise (в т.ч. ConvRot),
convrot_w4a4 (int4), asym_w4a8_int8 (int4 + codebook), w6a8_int8 — и любые новые форматы, которые
понимает установленная ComfyUI: квантованный тензор собирается РОДНЫМ кодом ComfyUI
(``comfy.ops._load_quantized_module``) и деквантуется comfy-kitchen (eager-backend работает на любой GPU,
включая V100), затем каждый rank берёт свои строки Shard(0) и хранит их в выбранном dtype
(fp16 на Volta, bf16 на Ampere/Ada/Blackwell — RTX 5090 и т.п.).

Часть модуля без torch: чтение comfy_quant JSON и логические (распакованные) формы весов — их используют
ноды/конфиги до запуска workers.
"""
import json

# Сколько бит на элемент в упакованном хранении (форма на диске [N, K*bits/8]).
PACKED_BITS = {"nvfp4": 4, "asym_w4a8_int8": 4, "w6a8_int8": 6}
QUANT_SUFFIX = ".comfy_quant"


def packed_bits(conf):
    fmt = conf.get("format")
    if fmt == "convrot_w4a4":
        params = conf.get("params") if isinstance(conf.get("params"), dict) else {}
        return 4 if conf.get("linear_dtype", params.get("linear_dtype", "int4")) == "int4" else 8
    return PACKED_BITS.get(fmt, 8)


def logical_shape(stored_shape, conf):
    """Форма исходного (деквантованного) веса по форме хранения и формату."""
    shape = list(stored_shape)
    bits = packed_bits(conf)
    if len(shape) == 2 and bits != 8:
        if (shape[1] * 8) % bits:
            raise ValueError(f"{conf.get('format')}: ширина {shape[1]} не соответствует {bits}-битной упаковке")
        shape[1] = shape[1] * 8 // bits
    return shape


def read_quant_json(path, data_start, desc):
    a, b = desc["data_offsets"]
    if desc["dtype"] != "U8" or b - a > 1 << 20:
        raise ValueError("Неверные comfy_quant metadata")
    with open(path, "rb") as f:
        f.seek(data_start + a)
        return json.loads(f.read(b - a))


def annotate_quantized(tensors, read_json):
    """tensors: имя -> desc (с ключом файла). Веса с comfy_quant получают логическую форму.

    desc["shape"] — логическая форма (для геометрии/LoRA/FSDP), desc["stored_shape"] — на диске,
    desc["comfy_quant"] — конфиг формата. Возвращает {модуль: конфиг}.
    """
    configs = {}
    for name in [n for n in tensors if n.endswith(QUANT_SUFFIX)]:
        module = name[:-len(QUANT_SUFFIX)]
        conf = read_json(name)
        if not isinstance(conf, dict) or not conf.get("format"):
            raise ValueError(f"comfy_quant без формата: {name}")
        configs[module] = conf
        weight = tensors.get(module + ".weight")
        if weight is not None:
            weight["stored_shape"] = list(weight["shape"])
            weight["shape"] = logical_shape(weight["shape"], conf)
            weight["comfy_quant"] = conf
    return configs


def module_entries(tensors, module):
    """Прямые тензоры модуля (weight + scales/codebook/...), кроме bias: имя -> ключ файла."""
    prefix = module + "."
    return {name: desc["key"] for name, desc in tensors.items()
            if name.startswith(prefix) and "." not in name[len(prefix):] and not name.endswith(".bias")}


def formats_summary(configs):
    out = {}
    for conf in configs.values():
        out[conf["format"]] = out.get(conf["format"], 0) + 1
    return out


# ------------------------------------------------------------------ torch side
def dequantize_module(handle, entries, module, conf, shape, device, dtype):
    """Полный деквантованный вес модуля родным кодом ComfyUI. entries: имя -> ключ файла."""
    import types
    import torch
    import comfy.ops
    prefix = module + "."
    state = {name: handle.get_tensor(key) for name, key in entries.items()}
    state.setdefault(prefix + "comfy_quant", torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8))
    loader = getattr(comfy.ops, "_load_quantized_module", None)
    fmt = conf.get("format")
    if loader is None and fmt in ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise") and not conf.get("convrot") \
            and not (conf.get("params") or {}).get("convrot"):
        # Старые ComfyUI без общего загрузчика: простые форматы — вручную (q * scale по строкам/тензору).
        weight = state[prefix + "weight"].to(device).float()
        scale = state.get(prefix + "weight_scale")
        if scale is not None:
            scale = scale.to(device).float()
            weight = weight * (scale.reshape(-1, *([1] * (weight.ndim - 1))) if scale.numel() > 1 else scale.reshape(()))
        return weight.to(dtype)
    if loader is None:
        raise RuntimeError("Эта версия ComfyUI не даёт comfy.ops._load_quantized_module: обновите ComfyUI "
                           f"для формата {conf.get('format')} ({module})")
    stand = types.SimpleNamespace(factory_kwargs={"device": device, "dtype": dtype}, _disabled_formats=set(),
                                  _orig_shape=tuple(shape), _full_precision_mm=False, _full_precision_mm_config=False,
                                  weight=None, quant_format=None, layout_type=None)
    loader(stand, lambda *args, **kwargs: None, state, prefix, {}, False, [], [], [])
    weight = stand.weight
    if weight is None:
        raise ValueError(f"{module}: ComfyUI не собрал квантованный вес ({conf.get('format')})")
    full = weight.dequantize() if hasattr(weight, "dequantize") else weight
    pre = state.get(prefix + "pre_quant_scale")
    if pre is not None:
        # ModelOpt AWQ: y = (x * s) Wq^T = x (Wq * s)^T — масштаб входов вносится в деквантованный вес.
        full = torch.as_tensor(full).to(device).float() * pre.to(device).float().reshape(1, -1)
    full = torch.as_tensor(full).to(device=device, dtype=dtype)
    if tuple(full.shape) != tuple(shape):
        raise ValueError(f"{module}: деквантованная форма {tuple(full.shape)} != {tuple(shape)} ({conf.get('format')})")
    if not torch.isfinite(full).all():
        raise FloatingPointError(f"{module}: не конечные значения после деквантования ({conf.get('format')})")
    return full


WEIGHT_FORMATS = ("dequantize", "as_file", "int8", "fp8", "mxfp8", "nvfp4", "int4")
COMPUTE_MODES = ("auto", "native", "dequantize")


def validate_quant_choice(weight_format, compute):
    if weight_format not in WEIGHT_FORMATS:
        raise ValueError(f"weight_format: {' / '.join(WEIGHT_FORMATS)}")
    if compute not in COMPUTE_MODES:
        raise ValueError(f"compute: {' / '.join(COMPUTE_MODES)}")


def resolve_weight_dtype(choice, selected):
    """auto: bf16, если все выбранные GPU >= sm80 (Ampere/Ada/Hopper/Blackwell), иначе fp16 (Volta/Turing)."""
    if choice in ("fp16", "bf16"):
        return choice
    if choice != "auto":
        raise ValueError("weight_dtype: auto / fp16 / bf16")
    caps = [tuple(d.get("capability") or (0, 0)) for d in selected]
    return "bf16" if caps and all(c >= (8, 0) for c in caps) else "fp16"
