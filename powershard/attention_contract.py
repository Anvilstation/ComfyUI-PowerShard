"""Внутренний контракт: Q BLHD, K/V BLHD; mask broadcast в B,H,Lq,Lk.

True в bool mask означает разрешённую связь. По умолчанию causal upper-left,
как SDPA. Возврат return_attn_probs: (output, logsumexp, probabilities после dropout).
Никакого автоматического определения layout по размерам.
"""
from dataclasses import dataclass
import math
import warnings
import torch
from torch.nn import functional as F


class UnsupportedAttention(RuntimeError):
    """Предсказуемое отсутствие семантики ДО вызова CUDA kernel, не CUDA error."""


@dataclass(frozen=True)
class AttentionOptions:
    softmax_scale: float | None = None
    mask: object = None
    causal: bool = False
    causal_alignment: str = "upper_left"
    dropout_p: float = 0.0
    window_size: tuple = (-1, -1)
    alibi_slopes: object = None
    deterministic: bool = False
    return_attn_probs: bool = False
    layout: str = "BLHD"


def validate(q, k, v, o):
    if o.layout != "BLHD" or any(x.ndim != 4 for x in (q,k,v)):
        raise ValueError("Явный layout BLHD требует четыре оси Q/K/V")
    b,lq,hq,d = q.shape
    bk,lk,hk,dk = k.shape
    bv,lv,hv,dv = v.shape
    if b != bk or b != bv or lk != lv or hk != hv or d != dk or not min(b,hq,hk,d,dv,lk) or hq % hk:
        raise ValueError("Несовместимые Q/K/V: batch, K/V length, heads, Q/K dim или GQA")
    if len({x.device for x in (q,k,v)}) != 1 or len({x.dtype for x in (q,k,v)}) != 1 or not q.is_floating_point():
        raise ValueError("Q/K/V должны иметь одинаковый floating dtype и device")
    if not 0 <= o.dropout_p < 1 or o.causal_alignment not in ("upper_left", "bottom_right"):
        raise ValueError("Некорректные dropout или causal_alignment")
    if len(o.window_size) != 2 or any(not isinstance(x,int) or x < -1 for x in o.window_size):
        raise ValueError("window_size: два целых >= -1")
    if o.softmax_scale is not None and not math.isfinite(o.softmax_scale):
        raise ValueError("softmax_scale должен быть конечным")
    if o.mask is not None:
        if o.mask.device != q.device or o.mask.dtype != torch.bool and not o.mask.is_floating_point():
            raise ValueError("mask: bool/floating на том же device")
        if torch.broadcast_shapes(o.mask.shape,(b,hq,lq,lk)) != (b,hq,lq,lk):
            raise ValueError("mask не broadcastable к B,H,Lq,Lk")
    if o.alibi_slopes is not None:
        if o.alibi_slopes.shape not in ((hq,), (b,hq)) or o.alibi_slopes.device != q.device:
            raise ValueError("ALiBi slopes: [Hq] или [B,Hq] на том же device")
    return b,lq,hq,d,lk,hk,dv


def expand_kv(q,k,v):
    repeats = q.shape[2] // k.shape[2]
    return (k.repeat_interleave(repeats,2),v.repeat_interleave(repeats,2)) if repeats != 1 else (k,v)


def bias_tile(q,k,o,a,b,c,d):
    lq,lk = q.shape[1],k.shape[1]
    offset = lk-lq if o.causal_alignment == "bottom_right" else 0
    i = torch.arange(a,b,device=q.device)[:,None]+offset
    j = torch.arange(c,d,device=q.device)[None,:]
    bias = torch.zeros((b-a,d-c),device=q.device,dtype=torch.float32)
    if o.causal:
        bias.masked_fill_(j>i,-torch.inf)
    left,right = o.window_size
    if left >= 0: bias.masked_fill_(j<i-left,-torch.inf)
    if right >= 0: bias.masked_fill_(j>i+right,-torch.inf)
    if o.alibi_slopes is not None:
        slopes = o.alibi_slopes.float()
        if slopes.ndim == 1: slopes = slopes[None]
        bias = bias - slopes[...,None,None]*(i-j).abs()
    if o.mask is not None:
        mask = torch.broadcast_to(o.mask,(q.shape[0],q.shape[2],lq,lk))[...,a:b,c:d]
        bias = bias + (torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,-torch.inf) if mask.dtype==torch.bool else mask.float())
    return bias


def math_attention(q,k,v,o,query_chunk=128,key_chunk=512,compute_fp16=False):
    b,lq,h,d,lk,hk,dv = validate(q,k,v,o)
    qh,kh,vh = (x.transpose(1,2) for x in (q,k,v))
    scale = d**-.5 if o.softmax_scale is None else o.softmax_scale
    output = torch.empty((b,h,lq,dv),device=q.device,dtype=q.dtype)
    lse = torch.empty((b,h,lq),device=q.device,dtype=torch.float32) if o.return_attn_probs else None
    if o.return_attn_probs and b*h*lq*lk*4 > 512*2**20:
        warnings.warn("return_attn_probs требует большой B*H*Lq*Lk tensor; обычный math использует tiles")
    probs = torch.empty((b,h,lq,lk),device=q.device,dtype=torch.float32) if o.return_attn_probs else None
    from .fp16_safe import scaled_matmul
    mm = scaled_matmul if compute_fp16 else lambda x,y: x.float() @ y.float()
    # Broadcast GQA groups at GEMM, never materialize the whole repeated KV.
    # Head order: contiguous groups of Hq/Hkv query heads share one KV head.
    def qk(qs,ks):
        if h == hk:
            return mm(qs,ks.transpose(-1,-2))
        return mm(qs.reshape(b,hk,h//hk,qs.shape[-2],d),ks.unsqueeze(2).transpose(-1,-2)).reshape(b,h,qs.shape[-2],ks.shape[-2])
    def pv(ps,vs):
        if h == hk:
            return mm(ps,vs)
        return mm(ps.reshape(b,hk,h//hk,ps.shape[-2],ps.shape[-1]),vs.unsqueeze(2)).reshape(b,h,ps.shape[-2],dv)
    for a in range(0,lq,query_chunk):
        z = min(a+query_chunk,lq)
        qs = qh[...,a:z,:]
        m = torch.full((b,h,z-a,1),-torch.inf,device=q.device)
        den = torch.zeros_like(m)
        acc = torch.zeros((b,h,z-a,dv),device=q.device)
        for c in range(0,lk,key_chunk):
            e = min(c+key_chunk,lk)
            scores = qk(qs,kh[...,c:e,:])*scale + bias_tile(q,k,o,a,z,c,e)
            new_m = torch.maximum(m,scores.amax(-1,keepdim=True))
            # Только fully masked rows имеют нулевой результат. NaN не скрывается.
            safe_m = torch.where(new_m == -torch.inf,0.,new_m)
            alpha = torch.exp(m-safe_m)
            p = torch.exp(scores-safe_m)
            pd = F.dropout(p,p=o.dropout_p,training=True) if o.dropout_p else p
            acc = acc*alpha + pv(pd,vh[...,c:e,:])
            den = den*alpha+p.sum(-1,keepdim=True)
            m = new_m
        output[...,a:z,:] = acc/torch.where(den==0,1.,den)
        if probs is not None:
            # Диагностический контракт: одна полная строка softmax, один dropout.
            scores = qk(qs,kh)*scale+bias_tile(q,k,o,a,z,0,lk)
            logsum = torch.logsumexp(scores,-1,keepdim=True)
            p = torch.exp(scores-torch.where(logsum == -torch.inf,0.,logsum))
            p = F.dropout(p,p=o.dropout_p,training=True) if o.dropout_p else p
            output[...,a:z,:] = pv(p,vh)
            probs[...,a:z,:],lse[...,a:z] = p,logsum.squeeze(-1)
    output = output.transpose(1,2).contiguous()
    return (output,lse,probs) if o.return_attn_probs else output


def sdpa_attention(q,k,v,o):
    validate(q,k,v,o)
    if o.return_attn_probs:
        raise UnsupportedAttention("SDPA не возвращает probabilities/LSE")
    if o.deterministic:
        raise UnsupportedAttention("SDPA automatic dispatch не гарантирует deterministic kernel; используйте math")
    if o.window_size != (-1,-1) or o.alibi_slopes is not None or (o.causal and o.causal_alignment != "upper_left") or (o.causal and o.mask is not None):
        raise UnsupportedAttention("Составной mask/window/ALiBi требует tiled math без большой dense mask")
    k,v = expand_kv(q,k,v)
    # SDPA accepts FP32 additive bias with half Q. Preserve its finite range.
    # Unsupported bias dtypes use the mathematically equivalent tiled path.
    mask = o.mask
    if mask is not None and mask.dtype not in (torch.bool, torch.float32, q.dtype):
        raise UnsupportedAttention("SDPA mask dtype: сохранить bias через math, без narrowing cast")
    args = dict(attn_mask=mask,dropout_p=o.dropout_p,is_causal=o.causal)
    if o.softmax_scale is not None: args["scale"] = o.softmax_scale
    # Никакого принудительного FLASH_ATTENTION-only context.
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),**args).transpose(1,2).contiguous()


def sdpa_tile_plan(q,k,workspace_bytes,query_chunk):
    b,lq,h,_ = q.shape
    lk = k.shape[1]
    # Conservative logits/probabilities/bias allowance, not kernel accounting.
    rows = max(1,min(lq,query_chunk,max(1,workspace_bytes//max(1,b*lk*16))))
    heads = max(1,min(h,workspace_bytes//max(1,b*rows*lk*16)))
    return dict(query_rows=rows,query_heads=heads,score_bytes=b*rows*heads*lk*4,
        temporary_estimate_bytes=b*rows*heads*lk*16,workspace_limit_bytes=workspace_bytes,
        minimum_tile_exceeds_budget=b*lk*16>workspace_bytes,
        note="SDPA automatic dispatch; estimate excludes full inputs/output, KV tile and kernel workspace")


def sdpa_attention_bounded(q,k,v,o,workspace_bytes,query_chunk=128):
    """Exact dense attention in Q/head tiles. K length is never truncated.

    Each tile carries its original query positions via bias_tile, so rectangular
    causal, window and ALiBi semantics survive. No full expanded GQA KV copy.
    Dropout is preserved (same distribution, not identical RNG consumption).
    """
    from dataclasses import replace
    b,lq,h,d,lk,hk,dv = validate(q,k,v,o)
    if o.return_attn_probs or o.deterministic:
        raise UnsupportedAttention("SDPA tiled не гарантирует deterministic и не возвращает LSE/probabilities")
    if o.mask is not None and o.mask.dtype not in (torch.bool,torch.float32,q.dtype):
        raise UnsupportedAttention("SDPA tiled сохраняет FP32 mask; для другого dtype нужен math")
    plan = sdpa_tile_plan(q,k,workspace_bytes,query_chunk)
    out = torch.empty((b,lq,h,dv),device=q.device,dtype=q.dtype)
    need_bias = o.mask is not None or o.causal or o.window_size != (-1,-1) or o.alibi_slopes is not None
    full_mask = None if o.mask is None else torch.broadcast_to(o.mask,(b,h,lq,lk))
    for i in range(0,h,plan["query_heads"]):
        j = min(i+plan["query_heads"],h)
        # A bounded KV tile, not Hq/Hkv copies of the entire sequence.
        indices = torch.arange(i,j,device=k.device)//(h//hk)
        ks,vs = k.index_select(2,indices),v.index_select(2,indices)
        qs = q[:,:,i:j]
        opts = replace(o,mask=None if full_mask is None else full_mask[:,i:j],
            alibi_slopes=None if o.alibi_slopes is None else o.alibi_slopes[...,i:j])
        for a in range(0,lq,plan["query_rows"]):
            z = min(a+plan["query_rows"],lq)
            bias = bias_tile(qs,ks,opts,a,z,0,lk) if need_bias else None
            args = dict(attn_mask=bias,dropout_p=o.dropout_p,is_causal=False)
            if o.softmax_scale is not None:args["scale"] = o.softmax_scale
            out[:,a:z,i:j] = F.scaled_dot_product_attention(qs[:,a:z].transpose(1,2),
                ks.transpose(1,2),vs.transpose(1,2),**args).transpose(1,2)
            del bias
        del ks,vs
    return out
