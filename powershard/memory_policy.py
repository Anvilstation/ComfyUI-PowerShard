"""Оценки для планирования, не admission check и не измеренная память активаций."""
import math


def allocator_budget(device):
    """Один CPU snapshot allocator на границе RPC, без sync каждого блока.

    Считаем только оценку целых свободных блоков native allocator. Фрагменты,
    private pools и внешние аллокации не превращают её в гарантию размещения.
    """
    import torch
    if device.type != "cuda":
        return dict(driver_free_bytes=2**30,planning_available_bytes=2**30,status="CPU_TEST_ESTIMATE")
    free,total=torch.cuda.mem_get_info(device)
    reserved,allocated=torch.cuda.memory_reserved(device),torch.cuda.memory_allocated(device)
    backend=torch.cuda.memory.get_allocator_backend()
    split=torch.cuda.memory_stats(device).get("inactive_split_bytes.all.current",0) if backend=="native" else None
    reusable=max(0,reserved-allocated-split) if split is not None else 0
    available=min(total,free+reusable)
    return dict(driver_free_bytes=free,cuda_reserved_bytes=reserved,cuda_allocated_bytes=allocated,
        inactive_split_bytes=split,reusable_cache_estimate_bytes=reusable,
        planning_available_bytes=available,allocator_backend=backend,
        note="Оценка, не admission gate: фрагментация/private pools/kernel workspace/другие процессы могут вызвать OOM")


def apply_workspace_policy(plan, config):
    """Same serializable policy in every rank; not an OOM admission check."""
    plan = dict(plan)
    plan["memory_settings"] = config.memory_settings()
    if config.min_vram:
        plan["mlp_budget_bytes"] = min(plan["mlp_budget_bytes"], config.workspace_mib*2**20)
        plan.update(effective_prefetch=0, mlp_mode_override="auto", allow_prepared_weights=False)
    return plan


def apply_linear_policy(net, config):
    # Instance attributes only, before FSDP. Never cast storage/quant metadata.
    if not config.min_vram:
        return
    from .operations import Linear
    for module in net.modules():
        if isinstance(module, Linear):
            module._ps_output_rows = config.dequant_rows


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
        if chunk >= 256:
            chunk = chunk // 256 * 256
        reason = "auto: локальные tokens, FP32 workspace, reserve, активные веса/prefetch"
    else:
        raise ValueError("Неизвестная MLP policy")
    return dict(mode=mode, requested_tokens=requested, effective_tokens=min(chunk, max(1,tokens)),
                local_tokens=tokens, chunks=math.ceil(tokens/chunk), workspace_bytes_per_token=row_bytes,
                estimated_chunk_workspace_bytes=min(tokens,chunk)*row_bytes,
                budget_bytes=budget_bytes, prepared_weight_bytes=prepared_bytes, reason=reason,
                estimate_not_measurement=True)


def plan_forward(free_bytes, reserve_bytes, activation_bytes, communication_bytes, group_bytes, prefetch_limit, policy):
    after_activations = max(0, free_bytes-reserve_bytes-activation_bytes-communication_bytes)
    effective = prefetch_limit
    if policy == "auto":
        effective = max(0,min(prefetch_limit,after_activations//max(1,group_bytes)-1))
    workspace = max(0,after_activations-(1+effective)*group_bytes)
    return dict(policy=policy, free_at_boundary=free_bytes, reserve_bytes=reserve_bytes,
        activation_estimate_bytes=activation_bytes, communication_estimate_bytes=communication_bytes,
        active_group_estimate_bytes=group_bytes, requested_prefetch=prefetch_limit,
        effective_prefetch=effective, mlp_budget_bytes=workspace,
        unknown="allocator fragmentation, kernel workspace, external allocations; no OOM admission gate")
