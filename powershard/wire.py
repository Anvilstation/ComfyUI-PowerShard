"""Локальный JSON+safetensors протокол. Callable/custom objects запрещены до collectives."""
import json
import uuid
from .host_guard import strip_internal_wrappers
from pathlib import Path

ALLOWED_OPTIONS = {"sample_sigmas", "sigmas", "cond_or_uncond", "original_shape", "block_index",
                   "minimax_h3_sigma_shift_video", "minimax_h3_sigma_shift_audio",
                   "prefetch_dynamic_vbars", "sigmas_schedule", "uuids"}


def has_effect(value):
    if isinstance(value, dict):
        return any(has_effect(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_effect(v) for v in value)
    return value is not None


class StagedTensor:
    """Ссылка на тензор, уже записанный в run-stage каталоге.

    write_payload сериализует её как {"$tensor": key, "$dir": index};
    read_payload(worker) разрешает её через base_directory без повторного
    H2D/D2H копирования и без повторной записи conditioning на каждом шаге.
    """

    def __init__(self, key, index=0):
        self.key, self.index = key, index

    def __repr__(self):
        return f"StagedTensor({self.key!r}, dir={self.index})"


def validate_options(options):
    for key, value in options.items():
        if key == "wrappers":
            value = strip_internal_wrappers(value)
        if key not in ALLOWED_OPTIONS:
            if isinstance(value, (dict, list, tuple)) and not has_effect(value):
                continue
            raise ValueError(f"PowerShard не поддерживает transformer_options[{key!r}]. Удалите wrapper/patch до sampling")
        if key == "prefetch_dynamic_vbars" and value:
            raise ValueError("Dynamic VRAM для распределённого генератора запрещён")


def write_payload(directory, value):
    import torch
    from safetensors.torch import save_file
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    tensors = {}
    def encode(x):
        if isinstance(x, torch.Tensor):
            if x.layout != torch.strided or x.is_nested:
                raise TypeError("Передавайте распакованные плотные tensors")
            key = f"t{len(tensors)}"
            tensors[key] = x.detach().to("cpu").contiguous().clone()
            return {"$tensor": key}
        if isinstance(x, uuid.UUID):
            return str(x)
        if isinstance(x, StagedTensor):
            return {"$tensor": x.key, "$dir": x.index}
        if x is None or type(x) in (str, bool, int, float):
            return x
        if isinstance(x, (list, tuple)):
            return {"$tuple" if isinstance(x, tuple) else "$list": [encode(v) for v in x]}
        if isinstance(x, dict):
            if not all(type(k) is str for k in x):
                raise TypeError("Ключи wire-словаря должны быть строками")
            # PackedLayout каждый rank восстанавливает родным кодом из сохранённых payload fields.
            clean = {}
            for k, v in x.items():
                if k == "layout" and type(v).__name__ == "PackedLayout" and type(v).__module__ == "comfy.ldm.minimax.model":
                    continue
                if k == "transformer_options":
                    validate_options(v)
                    v = dict(v)
                    if "wrappers" in v:
                        v["wrappers"] = strip_internal_wrappers(v["wrappers"])
                clean[k] = encode(v)
            return {"$dict": clean}
        raise TypeError(f"Нельзя передать {type(x).__name__}; callback/patch не сериализуется")
    encoded = encode(value)
    # Полная валидация завершена прежде чем workers получают команду.
    (p / "tree.json").write_text(json.dumps(encoded, allow_nan=False), encoding="utf-8")
    if tensors:
        save_file(tensors, str(p / "tensors.safetensors"))
    else:
        # Метка "стейджинг без тяжёлых тензоров": read_payload не будет
        # искать отсутствующий файл (например, step-payloads поверх run-stage).
        (p / "tensors.safetensors").write_bytes(b"")


def read_payload(directory, device="cpu", base_directory=None, stage_cache=None,
                 cache_device=None, cache_limit_bytes=None):
    import torch
    from safetensors import safe_open
    p = Path(directory)
    cache_device = device if cache_device is None else cache_device
    if cache_limit_bytes is not None and cache_limit_bytes < 0:
        raise ValueError("cache_limit_bytes должен быть >= 0")
    tree = json.loads((p / "tree.json").read_text())
    # base_directory: run-stage с тяжёлыми conditioning тензорами. Ссылки
    # {"$tensor": key} относятся к step-payload (последний источник),
    # {"$tensor": key, "$dir": i} — к i-му источнику (0 = run-stage).
    sources = [p]
    if base_directory is not None and Path(base_directory) != p:
        sources.insert(0, Path(base_directory))
    from contextlib import ExitStack
    stack = ExitStack()
    readers = []
    try:
        for source in sources:
            tensor_file = source / "tensors.safetensors"
            readers.append(stack.enter_context(safe_open(str(tensor_file), framework="pt", device="cpu"))
                           if tensor_file.exists() and tensor_file.stat().st_size > 0 else None)
        if stage_cache is not None:
            base_file = sources[0] / "tensors.safetensors"
            stat = base_file.stat()
            generation = (str(base_file.resolve()),stat.st_dev,stat.st_ino,stat.st_mtime_ns,stat.st_size,
                          str(device),str(cache_device),cache_limit_bytes)
            if stage_cache.get("__generation__") != generation:
                stage_cache.clear()
                stage_cache["__generation__"] = generation
    except BaseException:
        stack.close()
        raise
    def resolve(index):
        if index < 0 or index >= len(readers) or readers[index] is None:
            raise ValueError(f"Staged tensor ссылается на payload #{index}, но доступно только {len(readers)} источников")
        return readers[index]
    def owned_transfer(value, target):
        moved = value.to(target)
        # A cross-device copy already owns storage. Clone only when .to was a
        # no-op; do not double the CUDA peak by cloning a fresh H2D result.
        return moved.clone() if moved is value else moved
    def keep(key, value):
        size = value.untyped_storage().nbytes()
        if cache_limit_bytes is not None:
            if size > cache_limit_bytes:
                return
            def entries():
                return [(k,v) for k,v in stage_cache.items() if isinstance(v,torch.Tensor)]
            while sum(v.untyped_storage().nbytes() for _,v in entries())+size > cache_limit_bytes:
                del stage_cache[entries()[0][0]]
        stage_cache[key] = value
    def decode(x):
        if not isinstance(x, dict):
            return x
        if "$tensor" in x:
            index = x.get("$dir", len(readers)-1)
            if stage_cache is not None and "$dir" in x:
                # Staged conditioning: disk-read + H2D один раз, дальше GPU-клон
                # из кэша (изоляция от in-place мутаций модели сохранена).
                cache_key = x["$tensor"]
                if cache_key in stage_cache:
                    value = stage_cache.pop(cache_key)
                    stage_cache[cache_key] = value  # LRU; metadata is excluded
                    return owned_transfer(value, device)
                # Own storage even for device=cpu; never keep an mmap of a
                # published stage file that may be replaced by another prompt.
                value = owned_transfer(resolve(index).get_tensor(x["$tensor"]), cache_device)
                keep(cache_key, value)
                return owned_transfer(value, device)
            # clone отключает зависимость результата CPU от mmap после удаления input dir.
            return owned_transfer(resolve(index).get_tensor(x["$tensor"]),device)
        if "$tuple" in x:
            return tuple(decode(v) for v in x["$tuple"])
        if "$list" in x:
            return [decode(v) for v in x["$list"]]
        return {k: decode(v) for k, v in x["$dict"].items()}
    try:
        return decode(tree)
    finally:
        stack.close()
