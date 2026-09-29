"""Worker-local mixed precision; математика scaling независима от GPU architecture.

Идеи FP32 residual/condition и power-of-two compensation: Amduraznak и
MiniMaxH3-FP16Safe (MIT, см. licenses). Не переносит их global monkey patches.
"""
import types
import torch
from torch.nn import functional as F


def power2_scale(ratio):
    # Ограничивается только показатель масштаба снизу нулём, НЕ активации.
    # Все операции остаются на device; нет .item()/Python bool(tensor).
    return torch.exp2(torch.ceil(torch.log2(torch.maximum(ratio, torch.ones_like(ratio)))))


def prepare_matmul(b):
    """Подготовить только активную матрицу. Возвращаемые tensors не переживают MLP."""
    bf = b.float()
    sb = power2_scale(bf.abs().amax(dim=(-2, -1), keepdim=True) / 16384.)
    bh = (bf / sb).half()
    bound_b = bh.float().abs().sum(dim=-2, keepdim=True).amax(dim=-1, keepdim=True)
    return bh, sb, bound_b


def prepared_matmul(a, prepared):
    af = a.float()
    bh, sb, bound_b = prepared
    max_a = af.abs().amax(dim=-1, keepdim=True)
    sa = power2_scale(torch.maximum(max_a / 16384., max_a * bound_b / 16384.))
    return torch.matmul((af / sa).half(), bh).float() * sa * sb


def scaled_matmul(a, b):
    """A[...,M,K] @ B[...,K,N] через half GEMM, FP32 output и компенсацию.

    |A_i B_j| <= max(abs(A_i))*sum(abs(B_j)). Выбираем степени 2 по этой
    верхней границе с запасом 2, чтобы half OUTPUT GEMM тоже не переполнился.
    Деление не коммутируется через SiLU/RMSNorm: они видят восстановленный FP32.
    """
    af, bf = a.float(), b.float()
    if af.shape[-2] == 0:
        return torch.empty(af.shape[:-1] + (bf.shape[-1],), device=a.device, dtype=torch.float32)
    return prepared_matmul(af, prepare_matmul(bf))


def safe_linear(x, weight, bias=None):
    vector = x.ndim == 1
    out = scaled_matmul(x.unsqueeze(0) if vector else x, weight.transpose(-1, -2))
    if vector:
        out = out.squeeze(0)
    # Bias добавляется ПОСЛЕ компенсации; f(x/s)*s неверно при ненулевом bias.
    return out if bias is None else out + bias.float()


class FiniteTracker:
    """Один CPU read на границе RPC, flags принадлежат конкретному worker model."""
    def __init__(self, debug=False):
        self.debug = debug
        self.labels = ["output"]
        self.flags = None

    def register(self, label):
        self.labels.append(label)
        return len(self.labels)-1

    def begin(self, device):
        self.flags = torch.zeros(len(self.labels), dtype=torch.bool, device=device)

    def observe(self, slot, value):
        if self.flags is not None and (slot == 0 or self.debug):
            self.flags[slot].logical_or_(~torch.isfinite(value).all())

    def finish(self, outputs):
        for x in outputs if isinstance(outputs, (tuple, list)) else [outputs]:
            self.observe(0, x)
        bad = self.flags.cpu().tolist()  # единственная deferred D2H проверка
        self.flags = None
        if any(bad):
            raise FloatingPointError("H3 FP16 Safe: non-finite после mixed precision: " +
                                     ", ".join(name for name, flag in zip(self.labels, bad) if flag))


def fp32_stream_forward(self, x, *args, **kwargs):
    # Сохраняем родную modulation/masks/optional-attention логику ComfyUI.
    out = self._ps_native_forward(x.float(), *args, **kwargs)
    self._ps_tracker.observe(self._ps_finite_slot, out)
    return out


def chunked_mlp_forward(self, x):
    from .memory_policy import mlp_plan
    from .operations import prepared_linears
    from .telemetry import region
    shape = x.shape
    rows = x.reshape(-1, shape[-1])
    policy = getattr(self, "_ps_mlp_policy", None)
    mode = policy.mlp_chunk_mode if policy else "manual"
    context = getattr(self, "_ps_memory_context", {})
    requested_mode = mode
    mode = context.get("mlp_mode_override", mode)
    budget = context.get("mlp_budget_bytes", 256 * 2**20)
    plan = mlp_plan(len(rows),shape[-1],self.fc2.in_features,mode,self._ps_mlp_chunk,budget)
    plan["requested_mode"] = requested_mode
    result = torch.empty((rows.shape[0], self.fc2.out_features), dtype=torch.float32, device=x.device)
    # Подготовка весов один раз на MLP, только если остаётся бюджет для chunks.
    # FSDP оборачивает block: внутри этого цикла НЕТ FSDP collectives.
    allowance = max(0,budget-plan["estimated_chunk_workspace_bytes"]-result.numel()*4)
    with region("H3_MLP_prepare_and_chunks"),prepared_linears((self.fc1,self.fc2),allowance,
            enabled=plan["chunks"]>1 and context.get("allow_prepared_weights",True)) as prep:
        plan.update(prep)
        for a in range(0, rows.shape[0], plan["effective_tokens"]):
            b = min(a+plan["effective_tokens"], rows.shape[0])
            gate, up = self.fc1(rows[a:b]).chunk(2, -1)
            # SiLU нелинейна: ей передаётся истинный масштаб, а не scaled gate.
            activation = F.silu(gate.float()) * up.float()
            result[a:b] = self.fc2(activation)
            del gate, up, activation
    self._ps_mlp_report = plan
    self._ps_tracker.observe(self._ps_finite_slot, result)
    return result.reshape(shape[:-1] + (self.fc2.out_features,))


def install_safe_operations(net, tracker, active):
    from .operations import Linear, Int8Linear
    for name, module in net.named_modules():
        module._ps_tracker = tracker
        module._ps_finite_slot = tracker.register(name or "model")
        if isinstance(module, (Linear, Int8Linear)):
            module._ps_safe = active
            original_dtype = getattr(module, "_ps_original_compute_dtype", module.weight.dtype)
            module._ps_fp32 = name == "condition_proj" or original_dtype == torch.float32


def wrap_safe_block(block, chunk_tokens, policy=None):
    if not hasattr(block, "_ps_native_forward"):
        block._ps_native_forward = block.forward
        block.forward = types.MethodType(fp32_stream_forward, block)
    block.mlp._ps_mlp_chunk = chunk_tokens
    block.mlp._ps_mlp_policy = policy
    block.mlp.forward = types.MethodType(chunked_mlp_forward, block.mlp)


def apply_fp16_safe(net, policy):
    from .source_guard import check_h3_instance
    check_h3_instance(net)
    fingerprint = policy.fingerprint()
    previous = getattr(net, "_ps_patch_fingerprint", None)
    if previous is not None:
        if previous != fingerprint:
            raise RuntimeError("Изменение worker patch требует новой сессии, а не patch поверх FSDP")
        return net._ps_tracker
    tracker = FiniteTracker(policy.debug_finite)
    net._ps_patch_fingerprint = fingerprint
    net._ps_safe_active = policy.active
    net._ps_tracker = tracker
    install_safe_operations(net, tracker, policy.active)
    if policy.active:
        # dtype attribute — dtype потока, НЕ .float() всех весов.
        net.dtype = torch.float32
        for block in list(net.blocks) + list(net.token_refiner.blocks):
            wrap_safe_block(block, policy.mlp_chunk_tokens, policy)
        net.condition_proj._ps_fp32 = True
    return tracker
