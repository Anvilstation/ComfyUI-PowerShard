"""Квантованные Linear в FSDP2: хранение в формате файла или квантование при загрузке, вычисление
native-ядрами comfy-kitchen или деквантованием. Для Wan, LTX, MiniMax H3 и distributed text encoder.

Хранение. Квантованный вес любого формата ComfyUI (int8_tensorwise/ConvRot, float8, mxfp8, nvfp4,
convrot_w4a4 (int4), asym_w4a8_int8, w6a8_int8, ...) — это QuantizedTensor: упакованные данные + Params
(scales, block scales, codebook, ...). Раскладка block scales зависит от формата/ядер (swizzle, padding),
поэтому PowerShard её не интерпретирует: все тензоры QuantizedTensor упаковываются в ОДИН байтовый
вектор (выравнивание 16 байт) — FSDP2 шардирует и собирает байты, а в forward QuantizedTensor
восстанавливается из собранного вектора без копирования (view) и используется родным кодом
ComfyUI/comfy-kitchen. Это работает для любого формата, который понимает установленная ComfyUI.

Вычисление (на модуль):
  native     — F.linear(QuantizedTensor) как в ComfyUI MixedPrecisionOps: fp8/nvfp4/mxfp8 квантуют вход
               и идут в scaled-mm ядра; int8/int4 (weight-only) — int8_linear / w4a8 / convrot ядра;
  dequantize — вес деквантуется в fp16/bf16 (или FP32 для text encoder) на время GEMM: экономит VRAM
               и передачу, работает на любой GPU (V100);
  auto       — native, если GPU и CUDA-backend comfy-kitchen поддерживают формат, иначе dequantize.
"""
import dataclasses
import math
import types
import warnings
import torch
from torch import nn
from torch.nn import functional as F

ALIGN = 16
# Доп. тензоры модуля из файла (ModelOpt/AWQ): pre_quant_scale умножает вход, input_scale — статический
# масштаб квантования входа. Хранятся в том же байтовом векторе (имя спецификации "x:<имя>").
EXTRAS = ("pre_quant_scale", "input_scale")
# Минимальная compute capability для native-ядер и операция comfy-kitchen CUDA backend (None: torch._scaled_mm).
NATIVE = {"float8_e4m3fn": ((8, 9), None), "float8_e5m2": ((8, 9), None),
          "mxfp8": ((10, 0), "scaled_mm_mxfp8"), "nvfp4": ((10, 0), "scaled_mm_nvfp4"),
          "int8_tensorwise": ((7, 5), "int8_linear"), "asym_w4a8_int8": ((7, 5), "w4a8_int8_linear"),
          "w6a8_int8": ((7, 5), "w4a8_int8_linear"), "convrot_w4a4": ((7, 5), "convrot_w4a4_linear")}


@dataclasses.dataclass
class QuantPlan:
    fmt: str
    layout: object
    params_cls: type
    fields: dict
    specs: list          # [(имя, dtype, shape, offset, nbytes)], первым — "_qdata"
    total: int
    source: dict         # kind=file|quantize, module, key(s), ...

    def signature(self):
        return [(n, str(d), list(s), o, b) for n, d, s, o, b in self.specs]


# ------------------------------------------------------------- QuantizedTensor <-> bytes
def comfy_quant():
    import comfy.quant_ops as q
    return q


def layout_name(fmt):
    algos = comfy_quant().QUANT_ALGOS
    if fmt not in algos:
        raise ValueError(f"Формат {fmt} не поддерживается установленной ComfyUI/comfy-kitchen: {sorted(algos)}")
    return algos[fmt]["comfy_tensor_layout"]


def components(qt, extras=None):
    tensors, fields = [("_qdata", qt._qdata)], {}
    for field in dataclasses.fields(qt._params):
        value = getattr(qt._params, field.name)
        if isinstance(value, torch.Tensor):
            tensors.append((field.name, value))
        else:
            fields[field.name] = value
    for name, value in sorted((extras or {}).items()):
        tensors.append(("x:" + name, value))
    return tensors, fields


def _same_field(a, b):
    if isinstance(a, torch.dtype) or isinstance(b, torch.dtype):
        return True  # orig_dtype задаётся при unpack
    return a == b if not isinstance(a, (tuple, list)) else tuple(a) == tuple(b)


def plan_from_quantized(qt, fmt, source, extras=None):
    tensors, fields = components(qt, extras)
    specs, offset = [], 0
    for name, tensor in tensors:
        nbytes = tensor.numel() * tensor.element_size()
        specs.append((name, tensor.dtype, tuple(tensor.shape), offset, nbytes))
        offset += math.ceil(nbytes / ALIGN) * ALIGN
    return QuantPlan(fmt, qt._layout_cls, type(qt._params), fields, specs, max(offset, ALIGN), source)


def pack(qt, plan, device, extras=None):
    tensors, fields = components(qt, extras)
    if [(n, t.dtype, tuple(t.shape)) for n, t in tensors] != [(n, d, s) for n, d, s, _, _ in plan.specs]:
        raise RuntimeError(f"Раскладка {plan.fmt} отличается от шаблона: {[(n, t.dtype, tuple(t.shape)) for n, t in tensors]}")
    differ = {k: (v, plan.fields.get(k)) for k, v in fields.items() if not _same_field(v, plan.fields.get(k))}
    if differ or fields.keys() != plan.fields.keys():
        raise RuntimeError(f"Параметры {plan.fmt} отличаются от шаблона (данные были бы декодированы неверно): {differ}")
    out = torch.zeros(plan.total, dtype=torch.uint8, device=device)
    for (name, tensor), (_, _, _, offset, nbytes) in zip(tensors, plan.specs):
        out[offset:offset + nbytes].copy_(tensor.detach().contiguous().reshape(-1).view(torch.uint8).to(device))
    return out


def unpack(full, plan, dtype=None, with_extras=False):
    """Собранный байтовый вектор -> QuantizedTensor (view, без копий) [+ extras]."""
    q = comfy_quant()
    values, extras = {}, {}
    for name, tensor_dtype, shape, offset, nbytes in plan.specs:
        view = full[offset:offset + nbytes].view(tensor_dtype).reshape(shape)
        if name.startswith("x:"):
            extras[name[2:]] = view
        else:
            values[name] = view
    fields = dict(plan.fields)
    if dtype is not None and "orig_dtype" in fields:
        fields["orig_dtype"] = dtype
    params = plan.params_cls(**fields, **{k: v for k, v in values.items() if k != "_qdata"})
    qt = q.QuantizedTensor(values["_qdata"], plan.layout, params)
    return (qt, extras) if with_extras else qt


def quantized_from_file(handle, entries, module, conf, shape, device, dtype, meta=False):
    """QuantizedTensor из файла родным загрузчиком ComfyUI. meta=True — только шаблон (формы) без чтения данных."""
    import json
    import comfy.ops
    prefix = module + "."
    if meta:
        state = {name: torch.empty(desc[1], dtype=desc[0], device="meta") for name, desc in entries.items()}
    else:
        state = {name: handle.get_tensor(key) for name, key in entries.items()}
    state[prefix + "comfy_quant"] = torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)
    loader = getattr(comfy.ops, "_load_quantized_module", None)
    if loader is None:
        raise RuntimeError("Эта ComfyUI не даёт comfy.ops._load_quantized_module: хранение в формате файла недоступно")
    stand = types.SimpleNamespace(factory_kwargs={"device": torch.device("meta") if meta else device, "dtype": dtype},
                                  _disabled_formats=set(), _orig_shape=tuple(shape), _full_precision_mm=False,
                                  _full_precision_mm_config=False, weight=None, quant_format=None, layout_type=None)
    loader(stand, lambda *a, **k: None, state, prefix, {}, False, [], [], [])
    if stand.weight is None or not hasattr(stand.weight, "_qdata"):
        raise ValueError(f"{module}: ComfyUI не собрала квантованный вес ({conf.get('format')})")
    return stand.weight


def quantized_from_float(weight, fmt):
    q = comfy_quant()
    if fmt in ("float8_e4m3fn", "float8_e5m2"):
        # Как ComfyUI set_weight: масштаб по максимуму (иначе scale=1 и мелкие веса уходят в субнормали fp8).
        return q.QuantizedTensor.from_float(weight, layout_name(fmt), scale="recalculate")
    return q.QuantizedTensor.from_float(weight, layout_name(fmt))


def requantize_like(template, weight, fmt):
    """Новый вес (после LoRA) в раскладке файла: ConvRot/codebook/group параметры сохраняются (как ComfyUI LoRA)."""
    if hasattr(template, "requantize_from_float"):
        return template.requantize_from_float(weight, scale="recalculate")
    return quantized_from_float(weight, fmt)


# ------------------------------------------------------------------ module
class QuantLinear(nn.Module):
    """Linear, чьи веса хранятся одним байтовым вектором QuantizedTensor (FSDP Shard(0) по байтам)."""

    def __init__(self, src, plan, compute_dtype, native, follow_input=False, full_precision=False):
        super().__init__()
        self.in_features, self.out_features = src.in_features, src.out_features
        self.qbytes = nn.Parameter(torch.empty(plan.total, dtype=torch.uint8, device="meta"), requires_grad=False)
        self.bias = src.bias
        self._ps_qplan = plan
        self._ps_compute_dtype = compute_dtype
        self._ps_native = native
        self._ps_follow_input = follow_input
        self._ps_full_precision = full_precision  # comfy_quant full_precision_matrix_mult: только dequantize
        for attr in ("_ps_tracker", "_ps_finite_slot", "_ps_safe", "_ps_fp32"):
            if hasattr(src, attr):
                setattr(self, attr, getattr(src, attr))

    def extra_repr(self):
        return f"{self.in_features}, {self.out_features}, format={self._ps_qplan.fmt}, native={self._ps_native}"

    def forward(self, x):
        plan = self._ps_qplan
        shape = x.shape
        if self._ps_follow_input and not self._ps_native:
            dtype = x.dtype if x.is_floating_point() else self._ps_compute_dtype
        else:
            dtype = self._ps_compute_dtype
        rows = x.reshape(-1, shape[-1]).to(dtype)
        bias = None if self.bias is None else self.bias.to(dtype)
        qt, extras = unpack(self.qbytes, plan, dtype, with_extras=True)
        pre = extras.get("pre_quant_scale")
        if pre is not None:  # ModelOpt AWQ smoothing, как ComfyUI MixedPrecisionOps
            rows = rows * pre.to(rows.dtype)
        # FP16 Safe (V100): scaled half GEMM с FP32 выходом; native ядра выдают fp16 и могут переполниться.
        safe = getattr(self, "_ps_safe", False) and dtype == torch.float16
        out = None
        if self._ps_native and not safe and not self._ps_full_precision:
            try:
                out = native_linear(rows, qt, bias, plan.fmt, extras.get("input_scale"))
            except torch.cuda.OutOfMemoryError:
                raise
            except (NotImplementedError, RuntimeError, TypeError, ValueError) as error:
                warnings.warn(f"PowerShard: native {plan.fmt} GEMM недоступен ({error}); модуль переходит на dequantize")
                self._ps_native = False
                out = None
        if out is None and safe:
            from .fp16_safe import safe_linear
            out = safe_linear(rows, qt.dequantize().to(dtype), bias, None)
        elif out is None:
            out = F.linear(rows, qt.dequantize().to(dtype), bias)
        out = out.reshape(shape[:-1] + (self.out_features,))
        tracker = getattr(self, "_ps_tracker", None)
        if tracker is not None:
            tracker.observe(self._ps_finite_slot, out)
        return out


def native_linear(rows, qt, bias, fmt, input_scale=None):
    """Как ComfyUI MixedPrecisionOps.Linear (inference): квантование входа или weight-only GEMM."""
    q = comfy_quant()
    quantize_input = q.QUANT_ALGOS.get(fmt, {}).get("quantize_input", True)
    if quantize_input:
        scale = None if input_scale is None else input_scale.to(rows.device)
        qx = q.QuantizedTensor.from_float(rows, layout_name(fmt), scale=scale)
        out = F.linear(qx, qt, bias)
    else:
        out = F.linear(rows, qt, bias)
    if isinstance(out, q.QuantizedTensor):
        out = out.dequantize()
    return out


def native_support(fmt, device):
    """(bool, причина): есть ли native ядро для формата на этой GPU."""
    q = comfy_quant()
    if device.type != "cuda":
        return False, "не CUDA"
    need, op = NATIVE.get(fmt, ((99, 0), None))
    cap = torch.cuda.get_device_capability(device)
    if cap < need:
        return False, f"sm{cap[0]}{cap[1]} < sm{need[0]}{need[1]} для {fmt}"
    if op is None:
        return True, f"torch scaled_mm sm{cap[0]}{cap[1]}"
    if not getattr(q, "_CK_AVAILABLE", False):
        return False, "comfy-kitchen не импортирован"
    import comfy_kitchen as ck
    cuda = ck.list_backends().get("cuda", {})
    if not cuda.get("available") or cuda.get("disabled"):
        return False, "CUDA backend comfy-kitchen недоступен (" + str(cuda.get("unavailable_reason") or "disabled") + \
               "; нужен torch cu130+)"
    if op not in (cuda.get("capabilities") or []):
        return False, f"в CUDA backend нет {op}"
    return True, f"comfy-kitchen {op}"


# --------------------------------------------------------------- installation
TARGETS = {"int8": "int8_tensorwise", "fp8": "float8_e4m3fn", "mxfp8": "mxfp8", "nvfp4": "nvfp4", "int4": "convrot_w4a4"}


def is_linear(module):
    weight = getattr(module, "weight", None)
    return isinstance(module, nn.Linear) or (hasattr(module, "in_features") and isinstance(weight, nn.Parameter)
                                             and weight.ndim == 2)


def fits_blocks(module, fmt):
    k, n = module.in_features, module.out_features
    if fmt in ("nvfp4", "mxfp8", "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8"):
        return k % 64 == 0 and n % 16 == 0 and n >= 64
    return n >= 16


def install_quant_linears(net, tensors, weight_format, compute, scope, compute_dtype, device, open_file=None,
                          name_map=None, follow_input=False):
    """Заменить Linear в scope на QuantLinear. tensors: имя модели -> desc (key, shape, stored_shape, comfy_quant, dtype).

    weight_format: dequantize | as_file | int8 | fp8 | mxfp8 | nvfp4 | int4. Возвращает отчёт.
    name_map: имя параметра модели -> (таблица tensors, имя в таблице) для моделей с несколькими файлами (TE).
    """
    report = dict(weight_format=weight_format, compute=compute, modules={}, skipped=0, stored_bytes=0)
    if weight_format == "dequantize":
        return report
    templates, natives = {}, {}
    for name, module in list(net.named_modules()):
        if not is_linear(module) or getattr(module, "_ps_fp32", False) or isinstance(module, QuantLinear):
            continue
        if getattr(module, "weight", None) is not None and module.weight.dtype == torch.float32:
            continue  # FP32 острова (AdaLN/timestep/caption) не квантуются
        if scope and not any(name.startswith(s) for s in scope):
            continue
        table, key = (name_map(name + ".weight") if name_map else (tensors, name + ".weight"))
        desc = table.get(key) if table is not None else None
        if desc is None:
            continue
        file_conf = desc.get("comfy_quant")
        if weight_format == "as_file":
            if file_conf is None:
                continue
            fmt, kind = file_conf["format"], "file"
        else:
            fmt = TARGETS[weight_format]
            kind = "file" if file_conf is not None and file_conf.get("format") == fmt else "quantize"
        if not fits_blocks(module, fmt):
            report["skipped"] += 1
            continue
        module_key = key[:-len(".weight")]
        shape = [module.out_features, module.in_features]
        extras = None
        if kind == "file":
            entries = {n: (getattr(torch, _torch_dtype(d["dtype"])), d.get("stored_shape", d["shape"]))
                       for n, d in _entries(table, module_key).items()}
            extras = {x: torch.empty(entries[module_key + "." + x][1], dtype=entries[module_key + "." + x][0],
                                     device="meta") for x in EXTRAS if module_key + "." + x in entries}
            signature = ("file", fmt, json_key(file_conf), tuple(shape),
                         tuple((n[len(module_key):], str(v[0]), tuple(v[1])) for n, v in sorted(entries.items())))
            if signature not in templates:
                try:
                    qt = quantized_from_file(None, entries, module_key, file_conf, shape, device, compute_dtype, meta=True)
                except Exception:
                    if open_file is None:
                        raise
                    handle = open_file(table, key)
                    real = {n: d["key"] for n, d in _entries(table, module_key).items()}
                    qt = quantized_from_file(handle, real, module_key, file_conf, shape, torch.device("cpu"),
                                             compute_dtype)
                templates[signature] = qt
            qt = templates[signature]
            source = dict(kind="file", table=table, module=module_key, conf=file_conf, key=key)
        else:
            signature = ("quantize", fmt, tuple(shape))
            if signature not in templates:
                dummy = torch.ones(shape, dtype=compute_dtype, device=device)
                templates[signature] = quantized_from_float(dummy, fmt)
                del dummy
            qt = templates[signature]
            source = dict(kind="quantize", table=table, module=module_key, key=key)
        plan = plan_from_quantized(qt, fmt, source, extras)
        if fmt not in natives:
            if compute == "native":
                natives[fmt] = (True, "выбрано native")
            elif compute == "dequantize":
                natives[fmt] = (False, "выбрано dequantize")
            else:
                natives[fmt] = native_support(fmt, device)
        parent_name, _, leaf = name.rpartition(".")
        parent = net.get_submodule(parent_name) if parent_name else net
        full_precision = bool(kind == "file" and file_conf.get("full_precision_matrix_mult"))
        setattr(parent, leaf, QuantLinear(module, plan, compute_dtype, natives[fmt][0], follow_input, full_precision))
        entry = report["modules"].setdefault(fmt, dict(count=0, native=natives[fmt][0], reason=natives[fmt][1],
                                                      source={}))
        entry["count"] += 1
        entry["source"][kind] = entry["source"].get(kind, 0) + 1
        report["stored_bytes"] += plan.total
    return report


def json_key(conf):
    import json
    return json.dumps(conf, sort_keys=True)


def _entries(table, module):
    prefix = module + "."
    return {n: d for n, d in table.items()
            if n.startswith(prefix) and "." not in n[len(prefix):] and not n.endswith(".bias")
            and not n.endswith(".comfy_quant")}


def _torch_dtype(code):
    from .native_te import SAFETENSORS_DTYPES
    return SAFETENSORS_DTYPES[code]


# ---------------------------------------------------------------- loading
def float_weight(handle, table, key, device, dtype=torch.float32):
    """Полный вес как float: plain / fp8(_scaled) / comfy_quant (деквантование родным кодом)."""
    from .quant_formats import dequantize_module, module_entries
    desc = table[key]
    module = key[:-len(".weight")]
    if desc.get("comfy_quant") is not None:
        return dequantize_module(handle, module_entries(table, module), module, desc["comfy_quant"], desc["shape"],
                                 device, dtype)
    weight = handle.get_tensor(desc["key"]).to(device)
    if desc["dtype"] in ("F8_E4M3", "F8_E5M2"):
        weight = weight.float()
        for suffix in (".scale_weight", ".weight_scale"):
            scale = table.get(module + suffix)
            if scale is not None and scale["dtype"] in ("F32", "F16", "BF16"):
                value = handle.get_tensor(scale["key"]).float().to(device)
                weight = weight * (value.reshape(-1, 1) if value.numel() > 1 else value.reshape(()))
                break
    return weight.to(dtype)


def build_bytes(module, handle, device, lora_delta=None):
    """Полный байтовый вектор QuantLinear на device; возвращает (bytes, имена использованных тензоров таблицы)."""
    plan = module._ps_qplan
    source = plan.source
    table, key, name = source["table"], source["key"], source["module"]
    used = set(_entries(table, name)) | {key}
    if name + ".comfy_quant" in table:
        used.add(name + ".comfy_quant")
    dtype = module._ps_compute_dtype
    shape = [module.out_features, module.in_features]
    delta = lora_delta(key, shape, device) if lora_delta else None
    extras = None
    if source["kind"] == "file":
        entries = {n: d["key"] for n, d in _entries(table, name).items()}
        qt = quantized_from_file(handle, entries, name, source["conf"], shape, device, dtype)
        extras = {x: handle.get_tensor(table[name + "." + x]["key"]).to(device) for x in EXTRAS
                  if name + "." + x in table}
        if delta is not None:
            # LoRA к истинному весу W = Wq * pre_quant_scale (по входам): в пространстве Wq это delta / s.
            weight = qt.dequantize().float()
            pre = extras.get("pre_quant_scale")
            weight = weight + (delta.float() / pre.float().reshape(1, -1) if pre is not None else delta.float())
            if not torch.isfinite(weight).all():
                raise FloatingPointError(f"{key}: не конечные веса после LoRA")
            qt = requantize_like(qt, weight.to(dtype), plan.fmt)
            del weight
    else:
        weight = float_weight(handle, table, key, device)
        for suffix in (".scale_weight", ".weight_scale"):
            if name + suffix in table:
                used.add(name + suffix)
        if delta is not None:
            weight = weight + delta
        if not torch.isfinite(weight).all():
            raise FloatingPointError(f"{key}: не конечные веса до квантования")
        qt = quantized_from_float(weight.to(dtype), plan.fmt)
        del weight
    return pack(qt, plan, device, extras), used
