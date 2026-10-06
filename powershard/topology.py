"""Linux NUMA/NVLink inventory; affinity меняется только у собственного worker."""
import os
from pathlib import Path
import shutil
import subprocess
import warnings


def cpu_list(value):
    cpus = set()
    for part in value.strip().split(","):
        if not part:
            continue
        ends = part.split("-")
        cpus.update(range(int(ends[0]), int(ends[-1])+1))
    return sorted(cpus)


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def gpu_locality(ident):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"], text=True, timeout=10)
    bdf = next((line.split(",")[1].strip().lower() for line in raw.splitlines() if line.split(",")[0].strip()==ident), None)
    if bdf is None:
        raise ValueError(f"GPU UUID отсутствует в inventory: {ident}")
    # nvidia-smi domain бывает 8 hex digits, sysfs использует 4.
    domain, bus, function = bdf.split(":")
    pci = Path("/sys/bus/pci/devices") / f"{int(domain,16):04x}:{bus}:{function}"
    node = read_text(pci/"numa_node")
    local = read_text(pci/"local_cpulist") or ""
    allowed = set(os.sched_getaffinity(0))
    cpus = sorted(set(cpu_list(local)) & allowed)
    return dict(uuid=ident, pci_bus_id=bdf, numa_node=int(node) if node else -1,
                local_cpus=cpu_list(local), permitted_local_cpus=cpus, inherited_affinity=sorted(allowed))


def numa_launch_prefix(ident, policy):
    if policy != "bind":
        return []
    try:
        info = gpu_locality(ident)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        warnings.warn(f"NUMA locality недоступна: {error}; используем inherited affinity")
        return []
    executable = shutil.which("numactl")
    if info["numa_node"] < 0 or not info["permitted_local_cpus"]:
        warnings.warn("NUMA bind недоступен: нет locality; запуск без принудительной memory binding")
        return []
    if executable is not None:
        return [executable, "--physcpubind="+",".join(map(str,info["permitted_local_cpus"])),
                "--membind="+str(info["numa_node"])]
    # numactl-бинарника нет — membind применит сам worker через libnuma
    # (configure_worker_affinity). CPU affinity и так ставится in-process.
    return []


def _libnuma_bind(node):
    """In-process membind + run_on_node через libnuma без numactl-бинарника."""
    try:
        import ctypes
        lib = ctypes.CDLL("libnuma.so.1")
        if lib.numa_available() < 0:
            return False, "numa_available() < 0"
        lib.numa_bitmask_alloc.restype = ctypes.c_void_p
        lib.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        lib.numa_bitmask_free.argtypes = [ctypes.c_void_p]
        lib.numa_set_membind.argtypes = [ctypes.c_void_p]
        mask = lib.numa_bitmask_alloc(lib.numa_num_configured_nodes())
        if not mask:
            return False, "numa_bitmask_alloc failed"
        try:
            lib.numa_bitmask_setbit(mask, node)
            lib.numa_set_membind(mask)
            lib.numa_run_on_node(node)
        finally:
            lib.numa_bitmask_free(mask)
        return True, "libnuma membind+run_on_node применены"
    except OSError as error:
        return False, f"libnuma недоступна: {error}"


def configure_worker_affinity(ident, policy):
    try:
        info = gpu_locality(ident)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        warnings.warn(f"NUMA inventory недоступен: {error}; inherited affinity")
        info = dict(uuid=ident, numa_node=-1, permitted_local_cpus=[], reason=str(error))
    info["policy"] = policy
    if policy in ("auto", "bind") and info["numa_node"] >= 0 and info["permitted_local_cpus"]:
        try:
            os.sched_setaffinity(0, info["permitted_local_cpus"])
            info["applied"] = True
        except OSError as error:
            warnings.warn(f"CPU affinity не применена: {error}; сохраняем inherited affinity")
            info.update(applied=False,reason=str(error))
    else:
        info["applied"] = False
    if policy == "bind" and info["numa_node"] >= 0:
        # membind через libnuma, когда numactl-бинарника нет; с numactl его
        # сделал launch-prefix до старта процесса, здесь идемпотентный повтор
        # для гарантии (numa_set_membind можно вызывать повторно).
        ok, note = _libnuma_bind(info["numa_node"])
        info["membind"] = {"applied": ok, "note": note}
    info["effective_affinity"] = sorted(os.sched_getaffinity(0))
    info["memory_policy"] = "requested membind; verify /proc/self/numa_maps" if policy=="bind" else "first-touch; explicit membind not applied"
    return info


def process_memory():
    result = {"pid": os.getpid()}
    for file in ("/proc/self/status", "/proc/self/smaps_rollup"):
        raw = read_text(file) or ""
        for line in raw.splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmLck", "Rss", "Pss", "Private_Clean", "Private_Dirty", "Locked"}:
                result[key+"_bytes"] = int(value.strip().split()[0])*1024
    if "VmRSS_bytes" not in result:
        # Не-Linux (или /proc недоступен): RSS через resource/OSSpecific API,
        # чтобы callers и тесты получали согласованный ключ.
        try:
            import resource
            result["VmRSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        except Exception:
            try:
                import ctypes
                # Полная PROCESS_MEMORY_COUNTERS (10 полей): API проверяет
                # cb == sizeof(структуры) и возвращает ERROR_INSUFFICIENT_BUFFER
                # (122) на усечённой. HANDLE текущего процесса = (HANDLE)-1;
                # без явных argtypes ctypes портит его конверсией через c_int.
                class PMC(ctypes.Structure):
                    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
                pmc = PMC(); pmc.cb = ctypes.sizeof(pmc)
                psapi = ctypes.windll.psapi
                psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
                psapi.GetProcessMemoryInfo.restype = ctypes.c_int
                if psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1).value, ctypes.byref(pmc), pmc.cb):
                    result["VmRSS_bytes"] = pmc.WorkingSetSize
            except Exception:
                pass
    pages = {}
    for line in (read_text("/proc/self/numa_maps") or "").splitlines():
        for word in line.split():
            if word.startswith("N") and "=" in word and word[1:word.index("=")].isdigit():
                node, count = word.split("=")
                pages[node] = pages.get(node, 0)+int(count)
    result["numa_resident_pages_all_mappings"] = pages
    return result


def transfer_benchmark(device, mib=64, iterations=10):
    import time
    import torch
    rows = []
    count = mib*2**20//4
    for pinned in (False, True):
        host = torch.empty(count, dtype=torch.float32, pin_memory=pinned).fill_(.25)
        gpu = torch.empty(count, dtype=torch.float32, device=device)
        for direction in ("H2D", "D2H"):
            destination, source = (gpu, host) if direction=="H2D" else (host, gpu)
            for _ in range(2):
                destination.copy_(source, non_blocking=pinned)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(iterations):
                destination.copy_(source, non_blocking=pinned)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-start
            # CPU reads D2H only after synchronization. Проверка результата вне timing.
            assert bool((destination==.25).all().item())
            rows.append(dict(direction=direction, pinned=host.is_pinned(), bytes=count*4,
                             iterations=iterations, seconds=seconds, GB_per_s=count*4*iterations/seconds/1e9))
        del gpu, host
    return rows
