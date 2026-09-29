"""Выбор из CUDA-visible inventory, без CUDA initialization при импорте нод."""
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path


def current_inventory():
    import torch
    devices = []
    for index in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(index)
        uuid = getattr(p, "uuid", None)
        if uuid is None:
            import ctypes
            driver = ctypes.CDLL("libcuda.so.1")
            raw = (ctypes.c_ubyte * 16)()
            device_handle = ctypes.c_int()
            if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device_handle),index) or driver.cuDeviceGetUuid(raw, device_handle):
                raise RuntimeError("CUDA driver не предоставил UUID видимого устройства")
            from uuid import UUID
            uuid = "GPU-" + str(UUID(bytes=bytes(raw)))
        uuid = str(uuid)
        if not uuid.startswith(("GPU-", "MIG-")):
            uuid = "GPU-" + uuid
        devices.append(dict(user_id=str(index), uuid=uuid, name=p.name,
                            total_memory=p.total_memory, capability=[p.major, p.minor]))
    return devices


def visible_inventory(timeout=30):
    # Если Comfy уже инициализировал CUDA, именно его mapping авторитетен.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return current_inventory()
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(Path(__file__).resolve().parents[1]),env.get('PYTHONPATH','')])
    completed = subprocess.run([sys.executable, "-m", "powershard.devices"],
                               capture_output=True, text=True, timeout=timeout,env=env)
    if completed.returncode:
        raise RuntimeError("CUDA inventory failed: " + completed.stderr[-2000:])
    return json.loads(completed.stdout)


def resolve_gpu_selection(ids, inventory=None):
    from .config import DistributedConfig
    ids = DistributedConfig(gpu_ids=ids).gpu_ids
    inventory = visible_inventory() if inventory is None else inventory
    by_id = {d["user_id"]: d for d in inventory}
    by_uuid = {d["uuid"]: d for d in inventory}
    chosen = list(by_id) if ids == ("all",) else ids
    result, seen = [], set()
    for ident in chosen:
        device = by_id.get(str(int(ident))) if ident.isdigit() else by_uuid.get(ident)
        if device is None:
            raise ValueError(f"GPU {ident} не существует среди CUDA-устройств исходного процесса; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
        if device["uuid"] in seen:
            warnings.warn(f"Повторный GPU {ident} ({device['uuid']}) пропущен", stacklevel=2)
            continue
        seen.add(device["uuid"])
        rank = len(result)
        result.append(dict(device, requested_id=ident, rank=rank, worker_cuda_index=rank))
    if not result:
        raise ValueError("Нет выбранных доступных CUDA GPU")
    return result


if __name__ == "__main__":
    print(json.dumps(current_inventory()))
