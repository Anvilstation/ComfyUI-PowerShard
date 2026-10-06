"""Isolated import/kernel/numerical/performance probe; never an inference path."""
import argparse
import json
import math
import statistics
import time


def capture_kernels(call):
    import torch
    from .telemetry import profiler_summary
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            call();torch.cuda.synchronize()
        result=profiler_summary(prof)
        result["status"]="CAPTURED" if result["cuda_kernel_count"] else "NO_CUDA_EVENTS"
        return result
    except Exception as error:
        if any(x in str(error).lower() for x in ("illegal memory", "out of memory", "device-side assert", "cuda error")):
            raise
        # Retain benchmark results; no CUDA calls after this failed attempt.
        return dict(status="UNAVAILABLE",reason=f"{type(error).__name__}: {error}")


def benchmark(provider, total=46535, world=5, heads=56, dim=128, mode="token", repeats=3):
    import torch
    from .telemetry import accelerator_inventory
    if not torch.cuda.is_available():
        return dict(status="NOT_RUN",reason="CUDA unavailable",accelerators=accelerator_inventory())
    device=torch.device("cuda:0")
    if provider=="triton":
        from .triton_smoke import add
        x=torch.arange(4096,device=device,dtype=torch.float32);y=x*3
        started=time.perf_counter();result=add(x,y);torch.cuda.synchronize()
        torch.testing.assert_close(result,x+y,atol=0,rtol=0)
        first_call_s=time.perf_counter()-started
        kernels=capture_kernels(lambda:add(x,y))
        return dict(status="PASS",import_result="PASS",kernel_result="PASS",numerical_result="PASS",
                    compile_and_first_call_s=first_call_s,kernels=kernels,
                    accelerators=accelerator_inventory(),h3_execution="NOT_USED_BY_POWERSHARD",
                    note="A pointwise JIT smoke does not certify Triton GEMM/attention or H3 acceleration")
    from .attention_probe import probe
    from .attention_providers import FlashProvider
    from .attention_contract import AttentionOptions,sdpa_attention
    local_heads=math.ceil(heads/world) if mode=="ulysses" else heads
    lq=total if mode=="ulysses" else math.ceil(total/world)
    certification=probe(provider,dim,local_heads)
    call=FlashProvider(provider) if provider in ("flash_attn","vllm_flash_attn") else sdpa_attention
    generator=torch.Generator(device=device).manual_seed(1729)
    q=torch.randn(1,lq,local_heads,dim,device=device,dtype=torch.float16,generator=generator)
    k,v=(torch.randn(1,total,local_heads,dim,device=device,dtype=torch.float16,generator=generator) for _ in range(2))
    options=AttentionOptions(softmax_scale=dim**-.5)
    started=time.perf_counter();result=call(q,k,v,options);torch.cuda.synchronize()
    warmup_s=time.perf_counter()-started
    # FP32 MATH reference for eight query rows with ALL keys. Do not allocate
    # a full Lq*Lk score matrix, and do not compare a provider with itself.
    from torch.nn.attention import sdpa_kernel,SDPBackend
    rows=torch.linspace(0,lq-1,min(8,lq),device=device).long()
    with sdpa_kernel(SDPBackend.MATH):
        reference=sdpa_attention(q[:,rows].float(),k.float(),v.float(),options)
    torch.testing.assert_close(result[:,rows].float(),reference,atol=.007,rtol=.007)
    maximum_error=float((result[:,rows].float()-reference).abs().max())
    del result,reference
    cuda_ms,wall_s=[],[]
    for _ in range(repeats):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();started=time.perf_counter();begin.record()
        result=call(q,k,v,options);end.record();torch.cuda.synchronize()
        wall_s.append(time.perf_counter()-started);cuda_ms.append(begin.elapsed_time(end))
        del result
    kernels=capture_kernels(lambda:call(q,k,v,options))
    return dict(status="PASS",import_result="PASS",kernel_result="PASS",numerical_result="PASS",
                provider=provider,geometry=dict(mode=mode,world=world,total=total,lq=lq,heads=local_heads,dim=dim),
                warmup_s=warmup_s,cuda_ms=cuda_ms,wall_s=wall_s,median_cuda_ms=statistics.median(cuda_ms),
                max_abs_reference=maximum_error,certification=certification,kernels=kernels,
                accelerators=accelerator_inventory(),note="Attention microbenchmark only; no FSDP/offload/MLP/NCCL time included")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("provider",choices=["flash_attn","vllm_flash_attn","sdpa","triton"])
    parser.add_argument("--total",type=int,default=46535)
    parser.add_argument("--world",type=int,default=5)
    parser.add_argument("--heads",type=int,default=56)
    parser.add_argument("--dim",type=int,default=128)
    parser.add_argument("--mode",choices=["token","ulysses"],default="token")
    parser.add_argument("--repeats",type=int,default=3)
    args=parser.parse_args()
    if min(args.total,args.world,args.heads,args.dim,args.repeats)<1:parser.error("sizes must be positive")
    try:result=benchmark(**vars(args))
    except Exception as error:result=dict(status="FAIL",reason=f"{type(error).__name__}: {error}",provider=args.provider)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result["status"]=="PASS" else 2


if __name__=="__main__":raise SystemExit(main())
