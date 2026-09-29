// Независимый CUDA probe: обычный malloc и cudaMallocManaged, не PyTorch allocator.
// Запускать через run_ats_probe.py: отдельный процесс на UUID, с timeout.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void touch(float* data, size_t count) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) data[i]+=1.f;
}
static void trial(bool managed, size_t bytes, int allowed) {
    if (!allowed) { std::printf("{\"status\":\"NOT_SUPPORTED\"}"); return; }
    const size_t count=bytes/sizeof(float);
    float* data=nullptr;
    if (managed) check(cudaMallocManaged(&data,bytes));
    else data=static_cast<float*>(std::malloc(bytes));
    if (!data) throw std::runtime_error("CPU allocation failed");
    for (size_t i=0;i<count;i++) data[i]=float(i%1024); // CPU first touch.
    size_t free_before,total,free_after;
    check(cudaMemGetInfo(&free_before,&total));
    auto a=std::chrono::steady_clock::now();
    touch<<<256,256>>>(data,count);check(cudaGetLastError());check(cudaDeviceSynchronize());
    auto b=std::chrono::steady_clock::now();
    constexpr int repeats=8;
    auto c=std::chrono::steady_clock::now();
    for (int i=0;i<repeats;i++) touch<<<256,256>>>(data,count);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    auto d=std::chrono::steady_clock::now();
    check(cudaMemGetInfo(&free_after,&total));
    for (size_t i=0;i<count;i++) if (data[i]!=float(i%1024)+1+repeats) throw std::runtime_error("GPU/CPU integrity mismatch");
    double cold=std::chrono::duration<double>(b-a).count();
    double warm=std::chrono::duration<double>(d-c).count()/repeats;
    std::printf("{\"status\":\"PASS\",\"bytes\":%zu,\"cold_s\":%.9f,\"warm_s\":%.9f,\"logical_read_write_GB_s\":%.6f,\"free_before\":%zu,\"free_after\":%zu,\"integrity\":\"PASS\"}",bytes,cold,warm,2.*bytes/warm/1e9,free_before,free_after);
    if (managed) check(cudaFree(data));else std::free(data);
}
int main(int argc,char** argv) {
    try {
        size_t mib=argc>1?std::stoull(argv[1]):64;
        if (!mib || mib>4096) throw std::runtime_error("Probe size must be 1..4096 MiB; this is not an oversubscription test");
        check(cudaSetDevice(0));
        cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,0));
        std::printf("{\"visible_ordinal\":0,\"pageableMemoryAccess\":%d,\"pageableMemoryAccessUsesHostPageTables\":%d,\"managedMemory\":%d,\"concurrentManagedAccess\":%d,\"system_malloc\":",prop.pageableMemoryAccess,prop.pageableMemoryAccessUsesHostPageTables,prop.managedMemory,prop.concurrentManagedAccess);
        trial(false,mib*1024*1024,prop.pageableMemoryAccess);
        std::printf(",\"managed\":");trial(true,mib*1024*1024,prop.managedMemory);
        std::printf(",\"scope\":\"Direct CPU-allocated pointer access and managed memory only; not H3, FSDP, transport identification or OOM guarantee\"}\n");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what());return 1; }
}
