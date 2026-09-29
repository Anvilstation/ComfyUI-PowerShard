"""Read-only диагностика. Может создать короткий CUDA context для свойств/P2P; ничего не устанавливает."""
import glob
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args, timeout=15):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def loaded_nccl():
    try:
        paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libnccl" in line})
        return {"paths": paths, "note_ru": "Пусто: библиотека ещё не загружена динамически или статически слинкована; повторить после NCCL probe" if not paths else "Из /proc/self/maps текущего процесса"}
    except OSError as e:
        return {"error": str(e)}


def diagnose(comfy_path=None):
    out = {"arch": platform.machine(), "os": platform.platform(), "python": sys.version, "executable": sys.executable,
           "glibc": platform.libc_ver(), "disk": shutil.disk_usage(os.getcwd())._asdict(), "packages": {}}
    for name in ("torch", "torchvision", "torchaudio", "safetensors", "tokenizers", "transformers", "ray", "xfuser", "kernels", "comfy-kitchen", "comfy-aimdo"):
        try:
            out["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out["packages"][name] = "NOT_INSTALLED"
    for name, args in {"gpu_inventory": ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,pci.bus_id", "--format=csv"],
                       "gpu_topology": ["nvidia-smi", "topo", "-m"], "gpu_p2p": ["nvidia-smi", "topo", "-p2p", "r"],
                       "nvlink": ["nvidia-smi", "nvlink", "--status"],
                       "nvlink_remote_pci": ["nvidia-smi", "nvlink", "--pcibusid"],
                       "numa_gpu_affinity": ["nvidia-smi", "topo", "-cpu"], "toolkit_nvcc": ["nvcc", "--version"],
                       "numa": ["numactl", "--hardware"], "cpu_topology": ["lscpu"],
                       "os_release": ["cat", "/etc/os-release"], "ram": ["cat", "/proc/meminfo"]}.items():
        out[name] = command(args)
    toolkit_paths = set(glob.glob("/usr/local/cuda*/version.json"))
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            toolkit_paths.add(str(Path(os.environ[env])/"version.json"))
    out["toolkit_version_files"] = {p: Path(p).read_text() for p in toolkit_paths if Path(p).is_file()}
    out["transport_overrides"] = {k: os.environ[k] for k in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "CUDA_VISIBLE_DEVICES") if k in os.environ}
    from .topology import gpu_locality, process_memory
    out["process_memory"] = process_memory()
    from .web_api import provider_inventory
    out["attention_providers"] = provider_inventory()
    from .runtime import _SESSIONS
    out["powershard_sessions"] = [dict(running=s.running,role=s.role,role_options=s.role_options,idle_on_cpu=s.idle_on_cpu,config=s.config.to_dict(),patch=s.patch.to_dict(),
        selected_devices=s.selected_devices,world_size=len(s.selected_devices),attention_policy=s.attention_policy,
        fingerprint=getattr(s,"fingerprint",None),last_command=s.history[-1] if s.history else None) for s in list(_SESSIONS)]
    if "torch" in sys.modules:
        from .conditioning_cache import _CACHES
        out["conditioning_caches"]=[c.report() for c in list(_CACHES)]
    inventory = command(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    if inventory.get("returncode") == 0:
        out["gpu_numa"] = []
        for uuid in inventory["stdout"].splitlines():
            try: out["gpu_numa"].append(gpu_locality(uuid.strip()))
            except Exception as e: out["gpu_numa"].append({"uuid":uuid,"error":str(e)})
    out["numa_nodes"] = {str(p):p.read_text().strip() for p in Path("/sys/devices/system/node").glob("node*/distance")}
    try:
        import torch
        out["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda, "config": torch.__config__.show(),
                        "cuda_available": torch.cuda.is_available(), "distributed": torch.distributed.is_available(),
                        "nccl_available": torch.distributed.is_available() and torch.distributed.is_nccl_available(),
                        "cuda_arch_list": torch.cuda.get_arch_list(), "gpus": []}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                out["torch"]["gpus"].append({"logical_id": i, "name": prop.name, "uuid": str(getattr(prop, "uuid", "unknown")),
                    "capability": torch.cuda.get_device_capability(i), "total_bytes": total, "free_bytes": free})
            out["torch"]["p2p_matrix"] = [[i != j and torch.cuda.can_device_access_peer(i,j) for j in range(torch.cuda.device_count())] for i in range(torch.cuda.device_count())]
            try:
                out["torch"]["nccl_version"] = torch.cuda.nccl.version()
            except Exception as e:
                out["torch"]["nccl_version"] = str(e)
        out["torch"]["loaded_nccl"] = loaded_nccl()
    except Exception as e:
        out["torch"] = {"error": str(e)}
    for p in (comfy_path, Path(__file__).resolve().parents[1]):
        if p:
            out.setdefault("repositories", {})[str(p)] = {"commit": command(["git", "-C", str(p), "rev-parse", "HEAD"]),
                "changes": command(["git", "-C", str(p), "status", "--porcelain"])}
    if comfy_path:
        ext = Path(comfy_path)/"custom_nodes"
        out["extensions"] = {p.name: command(["git", "-C", str(p), "rev-parse", "HEAD"]) for p in ext.iterdir() if p.is_dir()} if ext.is_dir() else {}
    return out
