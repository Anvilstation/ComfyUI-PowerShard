"""Точный dense attention с online softmax в FP32, без N×N allocation."""
import math
import torch
import torch.distributed as dist
from .config import shard_bounds


def exact_attention(q, k, v, query_chunk=128, key_chunk=512, compute_fp16=False):
    # [H, Q, D], [H, K, D]. Все keys участвуют в softmax.
    if q.ndim != 3 or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("Некорректные Q/K/V")
    if k.shape[1] == 0:
        raise ValueError("Пустой набор keys")
    out = torch.empty_like(q)
    scale = q.shape[-1] ** -0.5
    for i in range(0, q.shape[1], query_chunk):
        qs = q[:, i:i+query_chunk].float()
        m = torch.full(qs.shape[:2] + (1,), -float("inf"), device=q.device)
        den = torch.zeros_like(m)
        acc = torch.zeros_like(qs)
        for j in range(0, k.shape[1], key_chunk):
            if compute_fp16:
                from .fp16_safe import scaled_matmul
                scores = scaled_matmul(qs, k[:, j:j+key_chunk].transpose(-1, -2)) * scale
            else:
                scores = torch.matmul(qs, k[:, j:j+key_chunk].float().transpose(-1, -2)) * scale
            new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
            alpha = torch.exp(m - new_m)
            prob = torch.exp(scores - new_m)
            pv = scaled_matmul(prob, v[:, j:j+key_chunk]) if compute_fp16 else torch.matmul(prob, v[:, j:j+key_chunk].float())
            acc = acc * alpha + pv
            den = den * alpha + prob.sum(-1, keepdim=True)
            m = new_m
        out[:, i:i+query_chunk] = (acc / den).to(q.dtype)
    if not compute_fp16 and not torch.isfinite(out).all():
        raise FloatingPointError("NaN/Inf в attention; FP16-путь требует анализа диапазонов")
    return out


def gather_rows(x, total, group=None, dtype=None):
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    a, b = shard_bounds(total, rank, world)
    if x.shape[0] != b - a:
        raise ValueError("Неверное локальное число tokens")
    width = (total + world - 1) // world
    comm = x if dtype is None else x.to(dtype)
    # CUDA allocator reuses released storage. The old global dict retained
    # every video shape and returned aliases overwritten by later calls.
    padded = comm.new_zeros((width,) + tuple(x.shape[1:]))
    padded[:len(comm)].copy_(comm)
    output = comm.new_empty((width * world,) + tuple(x.shape[1:]))
    dist.all_gather_into_tensor(output, padded, group=group)
    result = output[:total]
    return result if dtype is None else result.to(x.dtype)


def ulysses_heads_to_sequence(tensors, total, heads, group=None):
    """Local rows/all heads -> global rows/head shard for ANY world size.

    Zero heads extend H to ceil(H/world)*world. Remove token padding BEFORE
    softmax and artificial heads after the inverse. Q/K/V are packed INSIDE
    each destination slice: [world, 3, width, h, D].
    """
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    width, h = math.ceil(total / world), math.ceil(heads / world)
    a, b = shard_bounds(total, rank, world)
    n, _, dim = tensors[0].shape
    if n != b-a or any(t.shape != (n, heads, dim) for t in tensors):
        raise ValueError("Ulysses local token/head geometry mismatch")
    parts = []
    for t in tensors:
        padded = t.new_zeros((width, h*world, dim))
        padded[:n, :heads].copy_(t)
        parts.append(padded.view(width, world, h, dim).transpose(0, 1))
    send = torch.stack(parts, dim=1).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    full = recv.permute(1, 0, 2, 3, 4).reshape(len(tensors), world*width, h, dim)
    return tuple(t[:total] for t in full), h*world, send.numel()*send.element_size()


def ulysses_sequence_to_heads(out, total, heads, group=None):
    """Inverse permutation: remove artificial heads, keep all real tokens."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    width = math.ceil(total / world)
    h, dim = out.shape[-2:]
    if out.shape[0] != total or h != math.ceil(heads / world):
        raise ValueError("Ulysses global token/head geometry mismatch")
    padded = out.new_zeros((world*width, h, dim))
    padded[:total].copy_(out)
    send = padded.view(world, width, h, dim)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    a, b = shard_bounds(total, rank, world)
    return recv.transpose(0, 1)[:b-a].reshape(b-a, world*h, dim)[:, :heads].contiguous()


def rope_split_half(q, table):
    # q:[S,H,D], table:[1,S,1,R/2,2,2], родная partial RoPE H3.
    half = table.shape[-3]
    pair = torch.stack((q[..., :half], q[..., half:2*half]), -1).float()
    rotated = torch.matmul(table[0].float(), pair.unsqueeze(-1)).squeeze(-1)
    return torch.cat((rotated[..., 0], rotated[..., 1], q[..., 2*half:].float()), -1).to(q.dtype)


def attention_forward(self, x, rope_freqs=None, transformer_options=None):
    n = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads*self.head_dim, dim=-1)
    q = self.q_norm(q.reshape(n, self.heads, self.head_dim))
    k = self.k_norm(k.reshape(n, self.heads, self.head_dim))
    v = v.reshape(n, self.heads, self.head_dim)
    if rope_freqs is not None:
        q, k = rope_split_half(q, rope_freqs), rope_split_half(k, rope_freqs)
    safe = getattr(self.qkv_proj, "_ps_safe", False)
    state = getattr(self, "_ps_sequence", None)
    sequence = state is not None and state.get("enabled", False)
    from .telemetry import region
    comm_dtype = None
    from .attention_contract import AttentionOptions
    scale, restore_v = self.head_dim**-.5, 1.
    scaled_wire = (sequence and safe and self._ps_attention.effective != "math"
                   and state["comm_dtype"] == "fp16")
    if scaled_wire:
        # Normalize BEFORE exchange. Q/K have checkpoint-derived RMS/RoPE
        # bounds. V uses one device-side global maximum so every rank has the
        # same scale, including uneven/padded head shards. Raw large FP32 V
        # is never blindly cast to half. Final residual gather stays FP32.
        sq, sk = self._ps_qk_scales
        q, k = (q.float()/sq).half(), (k.float()/sk).half()
        from .fp16_safe import power2_scale
        maximum = v.float().abs().amax()
        with region("sequence_V_scale_MAX"):
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        state["scale_collectives"] += 1
        restore_v = power2_scale(maximum/16384.)
        v = (v.float()/restore_v).half()
        scale *= sq*sk
    if sequence:
        if state["ulysses"]:
            with region("sequence_ulysses_QKV_all_to_all"):
                (q, k, v), padded_heads, exchanged_bytes = ulysses_heads_to_sequence(
                    (q, k, v), state["total"], self.heads)
            state["padded_heads"] = padded_heads
            state["ulysses_local_heads"] = q.shape[1]
            state["kv_collectives"] += 1
            state["kv_gathered_bytes"] += exchanged_bytes
        else:
            # Safe Linear restores FP32 V, possibly >65504. Never cast BEFORE
            # scale calculation: Inf cannot be repaired by later V scaling.
            dtype = None if safe else {"fp16": torch.float16, "fp32": torch.float32}[state["comm_dtype"]]
            with region("sequence_KV_all_gather"):
                kv = gather_rows(torch.stack((k, v), dim=1), state["total"], dtype=dtype)
                k, v = kv.unbind(1)
            comm_dtype = str(kv.dtype if dtype is None else dtype)
            state["kv_collectives"] += 1
            state["kv_gathered_bytes"] += kv.numel()*(kv.element_size() if dtype is None else 2 if dtype==torch.float16 else 4)
        state["effective_comm_dtype"] = str(k.dtype)
    if safe and self._ps_attention.effective != "math" and not scaled_wire:
        sq, sk = self._ps_qk_scales
        q, k = (q.float()/sq).half(), (k.float()/sk).half()
        from .fp16_safe import power2_scale
        restore_v = power2_scale(v.float().abs().amax()/16384.)
        v = (v.float()/restore_v).half()
        scale *= sq*sk
    out = self._ps_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        AttentionOptions(softmax_scale=scale), group=self._ps_attention_group, compute_fp16=safe)
    out = out.squeeze(0)  # BLHD dispatcher -> LHD exchange
    if safe and not scaled_wire:
        out = out.float()*restore_v
    if sequence and state["ulysses"]:
        exchanged_bytes = math.ceil(state["total"]/dist.get_world_size())*dist.get_world_size()*out.shape[1]*out.shape[2]*out.element_size()
        with region("sequence_ulysses_output_all_to_all"):
            out = ulysses_sequence_to_heads(out, state["total"], self.heads)
        state["out_collectives"] += 1
        comm_dtype = str(out.dtype)
        state["out_exchanged_bytes"] += exchanged_bytes
    if scaled_wire:
        out = out.float()*restore_v
    self._ps_shapes = dict(q=list(q.shape), k=list(k.shape), v=list(v.shape),
                           local_input=list(x.shape), communication_dtype=comm_dtype)
    return self.out_proj(out.reshape(n, self.heads*self.head_dim))


def install_attention(model, config, dispatcher=None):
    import types
    from comfy.ldm.minimax.model import Attention
    from .attention_policy import AttentionDispatcher
    dispatcher = dispatcher or AttentionDispatcher(config)
    for name,m in model.named_modules():
        if isinstance(m, Attention):
            m._ps_config = config
            m._ps_attention = dispatcher
            m._ps_attention_group = "token_refiner" if name.startswith("token_refiner.") else "dit"
            m._ps_qk_scales = (1.,1.)
            m.forward = types.MethodType(attention_forward, m)


def configure_safe_qk(model, checkpoint, cached_scales=None):
    """Полные маленькие norm vectors, не полные generator weights на CPU/GPU.

    |RMSNorm(x)_j| <= sqrt(D)*max|weight|; split-half RoPE добавляет не более
    sqrt(2). Используем 2*sqrt(D) и запас округления half weights. Scaling
    коммутируется только через dot-product, НЕ через norm/SiLU/softmax.
    """
    if cached_scales is not None:
        modules={name:m for name,m in model.named_modules() if hasattr(m,"_ps_attention")}
        if modules.keys()!=cached_scales.keys():
            raise RuntimeError("Phase cache QK metadata names differ")
        for name,module in modules.items():
            scales=cached_scales[name]
            if len(scales)!=2 or any(not math.isfinite(s) or s<=0 for s in scales):
                raise RuntimeError("Phase cache invalid QK scales: "+name)
            module._ps_qk_scales=tuple(scales)
        return
    from safetensors import safe_open
    with safe_open(str(checkpoint.path),framework="pt",device="cpu") as reader:
        for name,m in model.named_modules():
            if not hasattr(m,"_ps_attention"):continue
            scales=[]
            for norm in ("q_norm","k_norm"):
                weight=reader.get_tensor(name+"."+norm+".weight").float()
                if not torch.isfinite(weight).all():raise FloatingPointError("Non-finite QK norm weights")
                bound=2*math.sqrt(m.head_dim)*float(weight.abs().max())*1.01
                scales.append(2.**max(0,math.ceil(math.log2(max(1.,bound/128.)))))
            m._ps_qk_scales=tuple(scales)


def install_sequence(model, config=None):
    """Шардируем h перед block0; native masks/RoPE/layout остаются источником истины.

    token: contiguous rows + full K/V all-gather на блок (текущий контракт).
    ulysses: h тоже режется по tokens (та же экономия памяти/вычислений), но
    перед attention attention_forward делает all-to-all Q/K/V по head-оси:
    каждый rank получает полную последовательность ЧУЖОЙ head-группы, считает
    её точно и обратным all-to-all возвращает куски владельцам. После out-proj
    каждый rank снова держит только свои tokens. RoPE режется по tokens, как в
    token-режиме. Нулевые heads дополняют число heads до кратного world;
    они удаляются после обратного обмена. Финальный residual не сжимается.
    """
    import types
    from .config import DistributedConfig
    config = config or DistributedConfig()
    mode = config.sequence_mode
    state = {"total": 0, "comm_dtype": config.sequence_comm_dtype}
    for index, block in enumerate(model.blocks):
        original = block.forward
        def run(self, h, t_emb, segments, rope, transformer_options=None, attention=None,
                _i=index, _original=original):
            if attention is not None:
                raise ValueError("Подмена attention несовместима с sequence backend")
            if _i == 0:
                state["total"] = h.shape[0]
                state["kv_collectives"],state["kv_gathered_bytes"],state["out_collectives"] = 0,0,0
                state["scale_collectives"],state["out_exchanged_bytes"] = 0,0
                state["mode"] = mode
                world=dist.get_world_size()
                state["ulysses"] = mode == "ulysses"
                a_last,b_last=shard_bounds(state["total"],world-1,world)
                state["enabled"]=a_last<b_last and world>1
                if not state["enabled"] and not state.get("warned_empty"):
                    import warnings
                    warnings.warn("Sequence split создаёт пустой rank либо world=1; этот forward использует FSDP-only на всём выбранном наборе")
                    state["warned_empty"]=True
                if state["enabled"]:
                    a, b = shard_bounds(state["total"], dist.get_rank(),world)
                    h = h[a:b].clone()
                # ulysses тоже работает на локальных tokens: экономия памяти и
                # GEMM как в token-режиме; разрез по heads происходит только
                # внутри attention (all-to-all Q/K/V) и не меняет stream.
            if not state["enabled"]:
                return _original(h,t_emb,segments,rope,transformer_options=transformer_options or {})
            a, b = shard_bounds(state["total"], dist.get_rank(),dist.get_world_size())
            # Оба режима работают на локальных tokens: сегменты и RoPE режутся
            # одинаково. Разница только в attention: token = full K/V gather,
            # ulysses = all-to-all head-срезов (см. attention_forward).
            local = []
            for start, stop, row in segments:
                l, r = max(a, start), min(b, stop)
                if l < r:
                    if isinstance(row, torch.Tensor) and row.ndim > 0:
                        row = row[l-start:r-start]
                    local.append((l-a, r-a, row))
            result = _original(h, t_emb, local, rope[:, a:b], transformer_options=transformer_options or {})
            if _i == len(model.blocks)-1:
                result = gather_rows(result, state["total"])
            return result
        block.attn._ps_sequence = state
        block.forward = types.MethodType(run, block)
