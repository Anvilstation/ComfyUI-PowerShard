import torch
import torch.distributed as dist
from .fsdp_backend import sync


def collectives(device):
    world = dist.get_world_size()
    rank = dist.get_rank()
    a = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    if not torch.equal(a @ torch.eye(4, device=device), a):
        raise RuntimeError("CUDA compute test FAILED")
    x = torch.tensor([rank + 1.], device=device)
    dist.broadcast(x, src=world-1)
    if x.item() != world:
        raise RuntimeError("broadcast FAILED")
    x.fill_(rank + 1)
    result = torch.empty(world, device=device)
    dist.all_gather_into_tensor(result, x)
    if result.tolist() != list(range(1, world+1)):
        raise RuntimeError("all_gather FAILED")
    dist.all_reduce(x)
    if x.item() != world*(world+1)/2:
        raise RuntimeError("all_reduce FAILED")
    sync(device)
    return {"compute": "PASS", "broadcast": "PASS", "all_gather": "PASS", "all_reduce": "PASS",
            "backend": dist.get_backend(), "world_size": world, "device": str(device)}
