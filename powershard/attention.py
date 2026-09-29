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
    scale = None
    if dtype == torch.float16 and x.dtype != torch.float16:
        from .fp16_safe import power2_scale
        # Отдельная граница каждого trailing component: K не теряет диапазон
        # из-за крупного V. Малый MAX collective, без GPU->CPU scalar sync.
        bound = x.float().abs().amax(0, keepdim=True) if x.shape[0] else torch.zeros((1,)+x.shape[1:],device=x.device)
        dist.all_reduce(bound, op=dist.ReduceOp.MAX, group=group)
        scale = power2_scale(bound / 16384.)
        comm = (x.float() / scale).half()
    else:
        comm = x if dtype is None else x.to(dtype)
    # Владение результатом принадлежит caller. CUDA allocator может повторно
    # использовать освобождённую память; живые Tensor не перезаписываются.
    padded = torch.zeros((width,) + tuple(x.shape[1:]), dtype=comm.dtype, device=x.device)
    padded[:comm.shape[0]].copy_(comm)
    output = torch.empty((width * world,) + tuple(x.shape[1:]), dtype=comm.dtype, device=x.device)
    dist.all_gather_into_tensor(output, padded.contiguous(), group=group)
    result = output[:total]
    if scale is not None:
        return (result.float() * scale).to(x.dtype)
    if dtype is None:
        return result
    return result.to(x.dtype)


def rope_split_half(q, table):
    # q:[S,H,D], table:[1,S,1,R/2,2,2], родная partial RoPE H3.
    half = table.shape[-3]
    pair = torch.stack((q[..., :half], q[..., half:2*half]), -1).float()
    rotated = torch.matmul(table[0].float(), pair.unsqueeze(-1)).squeeze(-1)
    return torch.cat((rotated[..., 0], rotated[..., 1], q[..., 2*half:].float()), -1).to(q.dtype)


def attention_forward(self, x, rope_freqs=None, transformer_options=None):
    config = self._ps_config
    n = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    q = self.q_norm(q.reshape(n, self.heads, self.head_dim))
    k = self.k_norm(k.reshape(n, self.heads, self.head_dim))
    v = v.reshape(n, self.heads, self.head_dim)
    if rope_freqs is not None:
        q, k = rope_split_half(q, rope_freqs), rope_split_half(k, rope_freqs)
    state = getattr(self, "_ps_sequence", None)
    if state is not None and state.get("enabled", True):
        from .telemetry import region
        if state.get("ulysses"):
            with region("sequence_ulysses_all_to_all"):
                # Ulysses: вход n = ЛОКАЛЬНЫЕ tokens [a:b], все heads.
                # Прямой all-to-all по head-оси: отправляю world head-чанков,
                # получаю от каждого rank ЕГО head-чанк на ПОЛНОЙ
                # последовательности (padded width, как в gather_rows).
                # После обмена q/k/v: [width, h, D] — полная длина, свой
                # head-чанк. Attention считается точно на полной длине.
                world = dist.get_world_size()
                rank = dist.get_rank()
                h = self.heads // world
                total = state["total"]
                width = (total + world - 1) // world
                def split_heads(t):
                    # [n, H, D] -> [world, n, h, D]: chunk j = head-группа j.
                    padded = t.new_zeros((width, self.heads, self.head_dim))
                    padded[:n] = t
                    return padded.reshape(width, world, h, self.head_dim).transpose(0, 1).contiguous()
                q_part, k_part, v_part = split_heads(q), split_heads(k), split_heads(v)
                recv_shape = (world, width, h, self.head_dim)
                q_g = torch.zeros(recv_shape, dtype=q.dtype, device=q.device)
                # Leading axis is DESTINATION rank, not K/V.
                kv_in = torch.stack((k_part, v_part), dim=1).contiguous()
                kv_g = torch.empty_like(kv_in)
                dist.all_to_all_single(q_g, q_part)
                dist.all_to_all_single(kv_g, kv_in)
                # q_g: [world, width, h, D], world-ось = sender rank; sender-порядок
                # = порядок contiguous token-чанков, поэтому flatten первых двух
                # осей даёт padded глобальную последовательность (i,j) -> i*width+j.
                # Padding живёт только в хвосте последнего чанка -> [:total] чистит.
                q = q_g.reshape(width * world, h, self.head_dim)[:total]
                kv = kv_g.permute(1,0,2,3,4).reshape(2, width * world, h, self.head_dim)
                k, v = kv[0][:total], kv[1][:total]
            state["kv_collectives"] = state.get("kv_collectives",0)+2
            state["kv_gathered_bytes"] = state.get("kv_gathered_bytes",0)+q_part.numel()*q.element_size()+kv_in.numel()*kv_in.element_size()
            state["ulysses_local_heads"] = h
        else:
            with region("sequence_KV_all_gather"):
                # Одинаковый dtype/layout K/V: один collective вместо двух, без
                # изменения точности/байтов. Padding удаляется до attention.
                # FP16 exchange scales finite FP32 values BEFORE conversion.
                comm_dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(
                    state.get("comm_dtype"), None)
                kv = gather_rows(torch.stack((k,v),dim=1),state["total"],dtype=comm_dtype)
                k,v = kv.unbind(1)
            state["kv_collectives"] = state.get("kv_collectives",0)+1
            element_bytes = torch.empty((),dtype=comm_dtype or kv.dtype).element_size()
            state["kv_gathered_bytes"] = state.get("kv_gathered_bytes",0)+math.ceil(state["total"]/dist.get_world_size())*dist.get_world_size()*2*self.heads*self.head_dim*element_bytes
            if comm_dtype == torch.float16 and kv.dtype != torch.float16:
                state["scale_all_reduces"] = state.get("scale_all_reduces",0)+1
    from .attention_contract import AttentionOptions
    safe = getattr(self.qkv_proj, "_ps_safe", False)
    scale = self.head_dim**-.5
    restore_v = 1.
    if safe and self._ps_attention.effective != "math":
        # RMSNorm даёт аналитическую границу Q/K; коэффициенты получены один
        # раз из небольших norm weights ДО FSDP, не .item() в каждом блоке.
        sq,sk = self._ps_qk_scales
        q,k = (q.float()/sq).half(),(k.float()/sk).half()
        from .fp16_safe import power2_scale
        restore_v = power2_scale(v.float().abs().amax()/16384.)
        v = (v.float()/restore_v).half()
        scale *= sq*sk  # ровно одна компенсация QK, до softmax
    out = self._ps_attention(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),
        AttentionOptions(softmax_scale=scale),group=self._ps_attention_group,compute_fp16=safe)
    if safe:out = out.float()*restore_v
    state = getattr(self, "_ps_sequence", None)
    if state is not None and state.get("enabled", True) and state.get("ulysses"):
        with region("sequence_ulysses_all_to_all_out"):
            # Обратный обмен. out: [total, h, D] — выход моего head-чанка на
            # полной последовательности. Отправляю rank j мои строки для ЕГО
            # tokens (chunk j padded-глобальной последовательности), получаю
            # от каждого rank его head-чанк на МОИХ tokens; конкат по head-оси
            # даёт [n, H, D] — полный hidden локальных tokens для out_proj.
            world = dist.get_world_size()
            h = state.get("ulysses_local_heads", self.heads // world)
            total = state["total"]
            width = (total + world - 1) // world
            padded = torch.zeros(width * world, h, self.head_dim, dtype=out.dtype, device=out.device)
            padded[:total] = out.squeeze(0)
            send = padded.reshape(world, width, h, self.head_dim).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send)
            a, b = shard_bounds(total, dist.get_rank(), world)
            out = torch.cat([recv[j][:b - a] for j in range(world)], dim=1)
    self._ps_shapes = dict(q=list(q.shape),k=list(k.shape),v=list(v.shape),local_input=list(x.shape),
                          communication_dtype=str(comm_dtype or kv.dtype) if (state and state.get("enabled") and not state.get("ulysses")) else None)
    return self.out_proj(out.reshape(n, self.heads * self.head_dim))


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


def configure_safe_qk(model, checkpoint):
    """Полные маленькие norm vectors, не полные generator weights на CPU/GPU.

    |RMSNorm(x)_j| <= sqrt(D)*max|weight|; split-half RoPE добавляет не более
    sqrt(2). Используем 2*sqrt(D) и запас округления half weights. Scaling
    коммутируется только через dot-product, НЕ через norm/SiLU/softmax.
    """
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
    token-режиме. Требует heads % world == 0 (56 → 2/4/7/8/14/28/56);
    иначе автоматический откат в token-режим с warning.
    """
    import types
    from .config import DistributedConfig
    config = config or DistributedConfig()
    mode = config.sequence_mode
    comm_dtype = {"fp16": torch.float16, "fp32": torch.float32}[config.sequence_comm_dtype]
    state = {"total": 0, "comm_dtype": config.sequence_comm_dtype}
    for index, block in enumerate(model.blocks):
        original = block.forward
        def run(self, h, t_emb, segments, rope, transformer_options=None, attention=None,
                _i=index, _original=original):
            if attention is not None:
                raise ValueError("Подмена attention несовместима с sequence backend")
            if _i == 0:
                state["total"] = h.shape[0]
                state["kv_collectives"],state["kv_gathered_bytes"],state["scale_all_reduces"] = 0,0,0
                state["requested_mode"] = mode
                world=dist.get_world_size()
                heads = getattr(self.attn, "heads", 0)
                ulysses_ok = mode == "ulysses" and heads and heads % world == 0
                if mode == "ulysses" and not ulysses_ok and not state.get("warned"):
                    import warnings
                    warnings.warn(f"sequence_mode=ulysses требует heads % world == 0 (heads={heads}, world={world}); этот forward использует token-режим")
                    state["warned"]=True
                state["ulysses"] = ulysses_ok
                a_last,b_last=shard_bounds(state["total"],world-1,world)
                state["enabled"]=a_last<b_last and world>1
                state["mode"] = ("ulysses" if ulysses_ok else "token") if state["enabled"] else "fsdp2"
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
                result = gather_rows(result, state["total"], dtype=comm_dtype if state.get("ulysses") else None)
            return result
        block.attn._ps_sequence = state
        block.forward = types.MethodType(run, block)
