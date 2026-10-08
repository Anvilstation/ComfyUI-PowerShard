"""Real Unified Memory shards; scoped allocator, no global PyTorch replacement.

ATS needs actual hardware/driver attributes, not UVA or a copy benchmark.
Physical residency is controlled by CUDA and is NOT torch.memory_allocated().
"""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile


ATTRIBUTES = {
    "unified_addressing": 41,  # CU_DEVICE_ATTRIBUTE, not cudaRuntime enum (27)
    "managed_memory": 83,
    "pageable_memory_access": 88,
    "concurrent_managed_access": 89,
    "pageable_memory_access_uses_host_page_tables": 100,
}


def ats_worker_environment(environ):
    """MemPool requires native allocator; select it only in isolated ATS ranks.

    ComfyUI often exports cudaMallocAsync by default. Keep its host environment
    untouched, and preserve other allocator options (including list values).
    """
    import re
    result = dict(environ)
    settings = result.get("PYTORCH_ALLOC_CONF", result.get("PYTORCH_CUDA_ALLOC_CONF", ""))
    parts = [part for part in settings.split(",")
             if part.strip() and not re.match(r"\s*(backend|expandable_segments)\s*:", part)]
    value = ",".join(parts+["backend:native", "expandable_segments:False"])
    result["PYTORCH_ALLOC_CONF"] = result["PYTORCH_CUDA_ALLOC_CONF"] = value
    return result


def driver_api():
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    cuda.cuDeviceGetAttribute.restype = ctypes.c_int
    cuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
    cuda.cuPointerGetAttribute.restype = ctypes.c_int
    if cuda.cuInit(0):
        raise RuntimeError("ATS: cuInit failed")
    return cuda


def ats_capabilities(index=0):
    cuda = driver_api()
    result = {"device_index": index, "attributes": {}, "errors": {}}
    for name, attribute in ATTRIBUTES.items():
        value = ctypes.c_int()
        rc = cuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, index)
        result["attributes"][name] = value.value if rc == 0 else None
        if rc:
            result["errors"][name] = rc
    required = ("managed_memory", "concurrent_managed_access", "pageable_memory_access",
                "pageable_memory_access_uses_host_page_tables")
    result["ats_supported"] = all(result["attributes"][k] == 1 for k in required)
    result["status"] = "PASS" if result["ats_supported"] else "FAIL"
    return result


class ManagedShardPool:
    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.capabilities = ats_capabilities(self.device.index)
        if not self.capabilities["ats_supported"]:
            raise RuntimeError("ATS unavailable: hardware/driver attributes=" + str(self.capabilities))
        if not all(hasattr(torch.cuda, k) for k in ("MemPool", "use_mem_pool")) or not hasattr(torch.cuda.memory, "CUDAPluggableAllocator"):
            raise RuntimeError("ATS requires torch.cuda.MemPool/use_mem_pool/CUDAPluggableAllocator")
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise RuntimeError("ATS scoped pool requires native allocator; remove backend:cudaMallocAsync for this run")
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            raise RuntimeError("ATS requires a C++ compiler (Ubuntu: build-essential); nvcc is not needed")
        self._build = tempfile.TemporaryDirectory(prefix="powershard-ats-")
        source = Path(__file__).with_name("ats_allocator.cpp")
        library = Path(self._build.name) / "ats_allocator.so"
        run = subprocess.run([compiler, "-std=c++11", "-shared", "-fPIC", "-O2", "-pthread",
                              str(source), "-o", str(library), "-ldl"], capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError("ATS allocator build failed: " + run.stderr[-4000:])
        self._allocator = torch.cuda.memory.CUDAPluggableAllocator(str(library), "ps_ats_alloc", "ps_ats_free")
        self.pool = torch.cuda.MemPool(self._allocator.allocator())
        self._driver = driver_api()

    def scope(self):
        import torch
        return torch.cuda.use_mem_pool(self.pool, device=self.device)

    def is_managed(self, tensor):
        if not tensor.numel():
            return True
        value = ctypes.c_uint()
        rc = self._driver.cuPointerGetAttribute(ctypes.byref(value), 8, tensor.data_ptr())  # IS_MANAGED
        return rc == 0 and value.value == 1

    def assert_parameters(self, root):
        for name, param in root.named_parameters():
            local = param.to_local()
            if local.device.type != "cuda" or not self.is_managed(local):
                raise RuntimeError("ATS parameter lost managed allocation: " + name)

    def report(self):
        return dict(capabilities=self.capabilities, allocator="cuMemAllocManaged",
                    torch_allocator="native; expandable_segments=False; isolated ATS worker only",
                    preferred_location="CPU", active_groups="ordinary CUDA VRAM",
                    logical_bytes_not_physical_residency=True,
                    status="EXPERIMENTAL; requires AC922 acceptance")
