"""CPU counters/shapes без чтения CUDA tensors; trace ranges только по opt-in."""
from contextlib import nullcontext
import os


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
        unit.register_forward_pre_hook(before)

    def report(self):
        return dict(units=self.units,fsdp_group_materializations_estimate=sum(v["calls"]*v["parameter_group_count"] for v in self.units.values()),
            fsdp_full_parameter_bytes_estimate=sum(v["calls"]*v["unsharded_parameter_bytes"] for v in self.units.values()),
            note="Module calls and group sizes; not measured NCCL bytes/time. Use optional profiler for actual collectives/overlap.")
