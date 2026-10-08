// Worker-only managed allocator for PyTorch CUDAPluggableAllocator.
// Uses the CUDA Driver ABI via dlsym: no nvcc/toolkit headers required.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <mutex>

namespace {
using Ptr = unsigned long long;
using Alloc = int (*)(Ptr*, std::size_t, unsigned int);
using Free = int (*)(Ptr);
using Advise = int (*)(Ptr, std::size_t, int, int);
std::once_flag once;
void* library = nullptr;
Alloc allocate = nullptr;
Free release = nullptr;
Advise advise = nullptr;
void init() {
    library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "PowerShard ATS: %s\n", dlerror()); return; }
    allocate = reinterpret_cast<Alloc>(dlsym(library, "cuMemAllocManaged"));
    release = reinterpret_cast<Free>(dlsym(library, "cuMemFree_v2"));
    advise = reinterpret_cast<Advise>(dlsym(library, "cuMemAdvise"));
}
}

extern "C" void* ps_ats_alloc(std::size_t size, int device, void*) {
    std::call_once(once, init);
    if (!allocate || !release || !advise) return nullptr;
    Ptr ptr = 0;
    int rc = allocate(&ptr, size, 1); // CU_MEM_ATTACH_GLOBAL
    if (rc) { std::fprintf(stderr, "PowerShard ATS cuMemAllocManaged: %d\n", rc); return nullptr; }
    // Keep persistent shards in RAM when VRAM is pressured, while permitting
    // direct GPU access. Collected active FSDP groups use the NORMAL GPU pool.
    rc = advise(ptr, size, 3, -1); // SET_PREFERRED_LOCATION, CU_DEVICE_CPU
    if (!rc) rc = advise(ptr, size, 5, device); // SET_ACCESSED_BY
    if (rc) {
        std::fprintf(stderr, "PowerShard ATS cuMemAdvise: %d\n", rc);
        release(ptr);
        return nullptr;
    }
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(ptr));
}

extern "C" void ps_ats_free(void* ptr, std::size_t, int, void*) {
    std::call_once(once, init);
    if (release && ptr) {
        int rc = release(static_cast<Ptr>(reinterpret_cast<std::uintptr_t>(ptr)));
        if (rc) std::fprintf(stderr, "PowerShard ATS cuMemFree: %d\n", rc);
    }
}
