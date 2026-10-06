"""Оценки для планирования, не admission check и не измеренная память активаций."""
import math


def available_memory(device):
    """Driver-free + reusable allocator cache, not total reserved storage.

    Read at a synchronized RPC boundary, or use active_bytes inside a forward
    so blocks awaiting stream completion are not counted as reusable. This is
    an estimate: fragmentation/external allocations can still cause an OOM.
    """
    import torch
    if device.type != "cuda":
        return dict(free=2**30, total=2**30, reusable_cache_bytes=0,
                    available_bytes=2**30, allocator="cpu")
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    stats = torch.cuda.memory_stats(device)
    active = max(allocated, stats.get("active_bytes.all.current", allocated))
    reusable = max(0, reserved-active)
    return dict(free=free, total=total, allocated=allocated, reserved=reserved,
                active_bytes=active, reusable_cache_bytes=reusable,
                available_bytes=min(total, free+reusable),
                allocator=torch.cuda.memory.get_allocator_backend())


def mlp_budget(context, device):
    # Attention temporaries have died before MLP. At this point actual allocator
    # occupancy also includes expanded reference/keyframe tokens and weights.
    # No D2H tensor reads or collectives; ranks may choose different chunk sizes.
    if device.type != "cuda":
        return context.get("mlp_budget_bytes", 256*2**20), {}
    snapshot = available_memory(device)
    pending_prefetch = context.get("effective_prefetch", 0)*context.get("active_group_estimate_bytes", 0)
    budget = max(0, snapshot["available_bytes"]-context.get("reserve_bytes", 0)-pending_prefetch)
    return budget, dict(budget_source="live allocator at MLP entry",
                        reusable_cache_bytes=snapshot["reusable_cache_bytes"],
                        pending_prefetch_allowance_bytes=pending_prefetch)


def mlp_plan(tokens, hidden, expansion, mode, requested, budget_bytes=None, prepared_bytes=0):
    # FP32 residual/output, gate/up/SiLU/product + half operands и scales.
    row_bytes = 16 * expansion + 12 * hidden + 16
    available = max(0, int(budget_bytes or 0) - prepared_bytes)
    if mode == "off":
        chunk, reason = max(1, tokens), "off: полный локальный MLP; FP16 Safe сохранён"
    elif mode == "manual":
        chunk, reason = requested, "manual: пользовательский размер не уменьшался"
    elif mode == "auto":
        chunk = max(1, min(tokens, available // row_bytes))
        # Не округлять короткие/последние порции до нуля.
        if 256 <= chunk < tokens:
            chunk = chunk // 256 * 256
        # A pessimistic estimate must never turn 50 blocks into hundreds of
        # thousands of single-token GEMMs. Attempt a bounded minimum chunk;
        # a real allocation failure remains an explicit CUDA OOM.
        floor = min(max(1, tokens), max(1, requested), 256)
        floor_used = chunk < floor
        chunk = max(chunk, floor)
        reason = "auto: доступная память с повторным использованием CUDA cache"
        if floor_used:
            reason += "; оценка исчерпана, минимальный chunk вместо token=1"
    else:
        raise ValueError("Неизвестная MLP policy")
    return dict(mode=mode, requested_tokens=requested, effective_tokens=min(chunk, max(1,tokens)),
                local_tokens=tokens, chunks=math.ceil(tokens/chunk), workspace_bytes_per_token=row_bytes,
                estimated_chunk_workspace_bytes=min(tokens,chunk)*row_bytes,
                budget_bytes=budget_bytes, prepared_weight_bytes=prepared_bytes, reason=reason,
                estimate_not_measurement=True,
                estimate_exhausted=mode == "auto" and floor_used)


def plan_forward(free_bytes, reserve_bytes, activation_bytes, communication_bytes, group_bytes, prefetch_limit, policy,
                 reusable_bytes=0):
    available = free_bytes+max(0, reusable_bytes)
    after_activations = max(0, available-reserve_bytes-activation_bytes-communication_bytes)
    effective = prefetch_limit
    if policy == "auto":
        effective = max(0,min(prefetch_limit,after_activations//max(1,group_bytes)-1))
    workspace = max(0,after_activations-(1+effective)*group_bytes)
    reason = "manual: requested prefetch unchanged; OOM remains possible" if policy == "manual" else "auto: requested prefetch fits the estimate"
    if policy == "auto" and effective < prefetch_limit:
        reason = "auto: insufficient estimated headroom after reserve/activation/communication/current group"
    return dict(policy=policy, free_at_boundary=free_bytes, reserve_bytes=reserve_bytes,
        reusable_cache_bytes=max(0, reusable_bytes), available_at_boundary=available,
        activation_estimate_bytes=activation_bytes, communication_estimate_bytes=communication_bytes,
        active_group_estimate_bytes=group_bytes, requested_prefetch=prefetch_limit,
        effective_prefetch=effective, mlp_budget_bytes=workspace,
        prefetch_reason=reason, prefetch_reduced=effective < prefetch_limit,
        unknown="allocator fragmentation, kernel workspace, external allocations; no OOM admission gate")
