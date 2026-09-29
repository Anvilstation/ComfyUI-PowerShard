"""CUDA capabilities + explicit DMA benchmark, НЕ ATS weight allocation.

isolated_probe запускается ДО worker/NCCL/весов. Bandwidth не доказывает NVLink.
"""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

# CUdevice_attribute в CUDA Driver API 12.4. Return codes сохраняются отдельно.
ATTRIBUTES = {"UNIFIED_ADDRESSING": 41, "MANAGED_MEMORY": 83,
    "PAGEABLE_MEMORY_ACCESS": 88, "CONCURRENT_MANAGED_ACCESS": 89,
    "PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES": 100,
    "DIRECT_MANAGED_MEM_ACCESS_FROM_HOST": 101}


def driver_capabilities(index):
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes, cuda.cuInit.restype = [ctypes.c_uint], ctypes.c_int
    cuda.cuDeviceGet.argtypes, cuda.cuDeviceGet.restype = [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int
    cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    cuda.cuDeviceGetAttribute.restype = ctypes.c_int
    rc = cuda.cuInit(0)
    if rc: raise RuntimeError(f"cuInit return code {rc}")
    device = ctypes.c_int()
    rc = cuda.cuDeviceGet(ctypes.byref(device), index)
    if rc: raise RuntimeError(f"cuDeviceGet({index}) return code {rc}")
    result = {}
    for name, enum in ATTRIBUTES.items():
        value = ctypes.c_int()
        rc = cuda.cuDeviceGetAttribute(ctypes.byref(value), enum, device)
        result[name] = dict(enum=enum, return_code=rc, raw=value.value if rc == 0 else None,
                            status="READ" if rc == 0 else "UNAVAILABLE")
    return result


def transfer_benchmark(index, mib=64, repeats=20):
    import time
    import torch
    device = torch.device("cuda", index)
    result = {}
    for pinned in (False, True):
        source = torch.empty(mib*2**20//4, dtype=torch.float32, pin_memory=pinned).fill_(.375)
        returned = torch.empty_like(source, pin_memory=pinned)
        gpu = torch.empty_like(source, device=device)
        measurements = {}
        for direction, dst, src in (("H2D", gpu, source), ("D2H", returned, gpu)):
            for _ in range(3): dst.copy_(src, non_blocking=pinned)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(repeats): dst.copy_(src, non_blocking=pinned)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            measurements[direction] = dict(seconds=seconds, GB_s=source.numel()*4*repeats/seconds/1e9)
        if not torch.equal(returned, source): raise RuntimeError("Transfer integrity check failed")
        result["pinned" if pinned else "pageable"] = dict(status="PASS", integrity="PASS", **measurements)
    return result


def probe_ats(index=0, benchmark=True):
    attributes = driver_capabilities(index)
    uva = attributes["UNIFIED_ADDRESSING"]["raw"]
    host_tables = attributes["PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES"]["raw"]
    pageable = attributes["PAGEABLE_MEMORY_ACCESS"]["raw"]
    report = dict(status="DIAGNOSTIC_ONLY", visible_ordinal=index, attributes=attributes,
        unified_addressing=None if uva is None else bool(uva),
        host_page_tables="SUPPORTED" if host_tables == 1 and pageable == 1 else "NOT_CONFIRMED",
        ats_execution="NOT_IMPLEMENTED: application uses CPUOffloadPolicy explicit copies",
        transport="UNKNOWN: bandwidth alone does not identify NVLink/PCIe", transfer={"status":"NOT_RUN"})
    if benchmark: report["transfer"] = transfer_benchmark(index)
    return report


def isolated_probe(device, timeout=90):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    try:
        process = subprocess.run([sys.executable, "-m", "powershard.ats_probe", "0"],
            env=env, text=True, capture_output=True, timeout=timeout)
        result = json.loads(process.stdout) if process.returncode == 0 else dict(status="ERROR",reason=process.stderr[-1500:])
    except (subprocess.TimeoutExpired, ValueError, OSError) as error:
        result = dict(status="ERROR", reason=str(error))
    return dict(result, uuid=device["uuid"], application_path="CPUOffloadPolicy; diagnostic failure does not veto normal offload")


if __name__ == "__main__":
    try: result = probe_ats(int(sys.argv[1]) if len(sys.argv)>1 else 0)
    except Exception as error: result = dict(status="ERROR",reason=f"{type(error).__name__}: {error}")
    print(json.dumps(result))
