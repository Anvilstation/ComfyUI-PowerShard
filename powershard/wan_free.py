"""Выгрузка host-моделей ComfyUI (text encoder, CLIP vision, VAE) из VRAM перед sampling PowerShard."""
import gc


def _patcher(obj):
    return getattr(obj, "patcher", obj)


def _is_powershard(patcher):
    from .comfy_adapter import PowerShardPatcher
    from .qwen_adapter import QwenPatcher
    return isinstance(patcher, (PowerShardPatcher, QwenPatcher)) or hasattr(getattr(patcher, "model", None), "session")


def _gpu_state(devices):
    """Только переданные устройства: без создания CUDA context на GPU workers."""
    import torch
    out = {}
    for device in devices:
        if getattr(device, "type", None) == "cuda" and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info(device)
            out[str(device)] = dict(free_gib=round(free / 2**30, 2), total_gib=round(total / 2**30, 2))
    return out


def free_host_models(mode, models):
    """mode=selected — только переданные объекты (и их clones); all_native — все native модели ComfyUI."""
    import comfy.model_management as mm
    closed = []
    targets = set()
    for obj in models:
        # Distributed umT5 PowerShard: закрыть его workers (кэш conditioning на host остаётся).
        if hasattr(obj, "free") and hasattr(getattr(obj, "patcher", None), "session"):
            obj.free()
            closed.append(type(obj).__name__)
            continue
        patcher = _patcher(obj)
        targets.add(getattr(patcher, "clone_base_uuid", id(patcher)))
    unloaded, keep, devices = [], [], {str(mm.get_torch_device()): mm.get_torch_device()}
    for loaded in list(mm.current_loaded_models):
        patcher = loaded.model
        if patcher is None:
            keep.append(loaded)
            continue
        if mode == "all_native":
            drop = not _is_powershard(patcher)
        else:
            drop = getattr(patcher, "clone_base_uuid", id(patcher)) in targets
        if drop:
            unloaded.append(type(getattr(patcher, "model", patcher)).__name__)
            devices[str(loaded.device)] = loaded.device
        else:
            keep.append(loaded)
    before = _gpu_state(devices.values())
    if unloaded:
        for device in devices.values():
            mm.free_memory(1e30, device, keep)
    gc.collect()
    mm.soft_empty_cache(True)
    return dict(mode=mode, unloaded=unloaded, closed_powershard_encoders=closed, vram_before=before,
                vram_after=_gpu_state(devices.values()))
