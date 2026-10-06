"""CPU counters/shapes без чтения CUDA tensors; trace ranges только по opt-in."""
from contextlib import nullcontext
import os
import json
import time
from pathlib import Path


class RequestProgress:
    """Atomic CPU-only checkpoint that survives a killed/interrupted worker."""
    def __init__(self, path, min_interval_s=0.5, **identity):
        self.path = Path(path)
        self.identity = identity
        self.current = {}
        self.min_interval_s = min_interval_s
        self._last_write = 0.
        self._writes = 0
        self._io_s = 0.
        self._mlp_written = False

    def begin(self, sequence, command):
        self.current = dict(self.identity, sequence=sequence, command=command,
                            status="RUNNING", started_unix_s=time.time())
        self._last_write = 0.
        self._writes = 0
        self._io_s = 0.
        self._mlp_written = False
        self("stage", "input_transfer")

    def __call__(self, key, value):
        self.current[key] = value
        self.current["updated_unix_s"] = time.time()
        now = time.perf_counter()
        force = key in ("stage", "status", "error", "memory_plan") or key == "mlp_plan" and not self._mlp_written
        if not force and now-self._last_write < self.min_interval_s:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.current, ensure_ascii=False))
        temporary.replace(self.path)
        self._io_s += time.perf_counter()-now
        self._last_write = now
        self._writes += 1
        if key == "mlp_plan":
            self._mlp_written = True

    def report(self):
        return dict(atomic_writes=self._writes, write_wall_s=self._io_s,
                    min_interval_s=self.min_interval_s, no_cuda_reads=True)


def profiler_summary(prof):
    """Observed ATen operators and CUDA kernel names; not hardware counters."""
    from collections import Counter
    kernels, durations = Counter(), Counter()
    for event in prof.events():
        if str(getattr(event, "device_type", "")).endswith("CUDA"):
            kernels[event.name] += 1
            durations[event.name] += getattr(event, "device_time_total", 0.)
    averages = prof.key_averages()
    operators = {e.key: e.count for e in averages if any(x in e.key.lower()
                 for x in ("scaled_dot_product", "flash_attn", "flashattention", "efficient_attention"))}
    names = list(kernels)
    return dict(cuda_kernel_count=sum(kernels.values()), attention_operators=operators,
        fsdp_forward_prefetch_ranges=[dict(name=e.key,count=e.count,cpu_total_us=e.cpu_time_total)
            for e in averages if "FSDP::forward_prefetch" in e.key],
        attention_kernel_names=[n for n in names if any(x in n.lower() for x in ("flash", "fmha", "attention", "mha_fwd"))],
        triton_kernel_names=[n for n in names if "triton" in n.lower()],
        gemm_kernel_names=[n for n in names if any(x in n.lower() for x in ("gemm", "mma"))][:40],
        top_cuda_kernels=[dict(name=n,count=kernels[n],total_us=t) for n,t in durations.most_common(20)],
        note="Kernel names observed by profiler; duration sum ignores overlap; Tensor Core utilization needs hardware counters")


def accelerator_inventory():
    import importlib.metadata
    import platform
    import torch
    from .reporting import environment_identity
    packages = {}
    for name in ("flash-attn", "vllm-flash-attn", "triton", "power-torch-cuda124", "torch"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "NOT_INSTALLED"
    return dict(environment=environment_identity(), packages=packages,
        architecture=platform.machine(),
        matmul=dict(allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                    allow_fp16_accumulation=getattr(torch.backends.cuda.matmul,"allow_fp16_accumulation",None)),
        triton=dict(explicit_powershard_kernel_path=False, torch_compile=False,
                    execution="NOT_USED_BY_POWERSHARD",
                    note="Installed version does not prove execution. Custom attention providers may have their own kernels; inspect a CUDA trace."))


def region(name):
    if os.environ.get("POWERSHARD_PROFILE") == "1":
        import torch
        return torch.profiler.record_function("PowerShard/"+name)
    return nullcontext()


class ForwardLedger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.units = {}

    def attach(self, name, unit, groups):
        sizes=[sum(p._orig_size.numel()*p.sharded_param.element_size() for p in g.fsdp_params) for g in groups]
        def before(module,args):
            row=self.units.setdefault(name,dict(calls=0,parameter_group_count=len(sizes),unsharded_parameter_bytes=sum(sizes)))
            row["calls"]+=1
            row["input_shapes"]=[list(x.shape) for x in args if hasattr(x,"shape")]
            state=module._get_fsdp_state()
            targets=getattr(state,"_states_to_forward_prefetch",None)
            if targets is not None:
                row["configured_prefetch_target_states"]=len(targets)
        unit.register_forward_pre_hook(before)

    def report(self):
        return dict(units=self.units,fsdp_group_materializations_estimate=sum(v["calls"]*v["parameter_group_count"] for v in self.units.values()),
            fsdp_full_parameter_bytes_estimate=sum(v["calls"]*v["unsharded_parameter_bytes"] for v in self.units.values()),
            note="Module calls and group sizes; not measured NCCL bytes/time. Use optional profiler for actual collectives/overlap.")
