"""Блочно-разреженные Triton-ядра Veda на тензорах в порядке тайлов [N, H', D].

* `attend_int8` — ядро Veda-on-ComfyUI (veda_comfy/kernels/sage/sparse_int8.py, MIT), производное от
  SageAttention 1.0.6 (BSD-3-Clause): INT8 Q/K со scale на блок, softmax scale вложен в квантование Q, P и V
  в fp16. Изменение для PowerShard: P·V накапливается в FP32 (out_dtype по умолчанию) — при FP16 Safe V
  нормирован к 16384 и сумма 64 ключей в fp16 переполнилась бы; softmax scale передаётся явно (Safe делит q/k
  на степени двойки).
* `attend_fp16` — то же разреженное обхождение ключей, но Q·K в fp16 с FP32-накоплением (tl.dot) и меньшим
  блоком запросов (64): для GPU без INT8 tensor cores в Triton (V100 / sm70) и как запасной вариант.

Общий инвариант тайлинга: реальные строки тайла — префикс длины valid_count, поэтому паддинг маскируется
одним сравнением.
"""
import functools

import torch
import triton
import triton.language as tl

TILE = 128
KEY_BLOCK = 64
LOG2E = 1.4426950408889634


# ----------------------------------------------------------------- INT8 (sm80+)
@triton.jit
def _quantize_kernel(X, Xq, Scale, stride_xn, stride_xh, stride_sh,
                     pre_scale, D: tl.constexpr, BLK: tl.constexpr):
    blk = tl.program_id(0)
    head = tl.program_id(1)
    offs_n = blk * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    offset = head * stride_xh + offs_n[:, None] * stride_xn + offs_d[None, :]
    x = tl.load(X + offset).to(tl.float32) * pre_scale
    scale = tl.max(tl.abs(x)) / 127.0
    scale = tl.where(scale > 0.0, scale, 1.0)
    quantized = x / scale
    quantized += 0.5 * tl.where(quantized >= 0, 1, -1)
    tl.store(Xq + offset, quantized.to(tl.int8))
    tl.store(Scale + head * stride_sh + blk, scale)


@triton.jit
def _attention_int8_kernel(Q, K, V, Q_scale, K_scale, Index, Count, Valid, Out,
                           stride_n, stride_h, stride_vn, stride_vh,
                           stride_qs, stride_ks, stride_ih, stride_iq,
                           n_q_tiles, D: tl.constexpr, BLK: tl.constexpr,
                           BLOCK_M: tl.constexpr):
    query_tile = tl.program_id(0)
    head = tl.program_id(1)
    offs_m = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + head * stride_h + offs_m[:, None] * stride_n + offs_d[None, :])
    q_scale = tl.load(Q_scale + head * stride_qs + query_tile)
    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    count = tl.load(Count + head * n_q_tiles + query_tile)
    index_row = Index + head * stride_ih + query_tile * stride_iq
    for i in range(0, count):
        block = tl.load(index_row + i)
        start = block * BLK
        k = tl.load(K + head * stride_h + (start + offs_n)[None, :] * stride_n + offs_d[:, None])
        k_scale = tl.load(K_scale + head * stride_ks + block)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        qk = tl.where(offs_n[None, :] < tl.load(Valid + block), qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + head * stride_vh + (start + offs_n)[:, None] * stride_vn + offs_d[None, :])
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_ij
    l_i = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_i[:, None]
    tl.store(Out + head * stride_h + offs_m[:, None] * stride_n + offs_d[None, :], acc.to(Out.type.element_ty))


# ------------------------------------------------------------------ FP16 (sm70+)
@triton.jit
def _attention_fp16_kernel(Q, K, V, Index, Count, Valid, Out,
                           stride_qn, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh,
                           stride_on, stride_oh, stride_ih, stride_iq, n_q_tiles, qk_scale,
                           D: tl.constexpr, BLK: tl.constexpr, BLOCK_M: tl.constexpr, SUB: tl.constexpr):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    query_tile = pid // SUB
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + head * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :])
    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    count = tl.load(Count + head * n_q_tiles + query_tile)
    index_row = Index + head * stride_ih + query_tile * stride_iq
    for i in range(0, count):
        block = tl.load(index_row + i)
        start = block * BLK
        k = tl.load(K + head * stride_kh + (start + offs_n)[None, :] * stride_kn + offs_d[:, None])
        qk = tl.dot(q, k) * qk_scale                       # FP32-накопление: |q·k| не ограничен fp16
        qk = tl.where(offs_n[None, :] < tl.load(Valid + block), qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + head * stride_vh + (start + offs_n)[:, None] * stride_vn + offs_d[None, :])
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_ij
    l_i = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_i[:, None]
    tl.store(Out + head * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :], acc.to(Out.type.element_ty))


# ----------------------------------------------------------------------- host
def quantize(x, block, pre_scale=1.0):
    slots, heads, dim = x.shape
    out = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((heads, slots // block), dtype=torch.float32, device=x.device)
    _quantize_kernel[(slots // block, heads)](x, out, scale, x.stride(0), x.stride(1), scale.stride(0),
                                              pre_scale=pre_scale, D=dim, BLK=block, num_warps=4)
    return out, scale


def key_blocks(index, count, valid_count, block=KEY_BLOCK):
    """Список оставленных тайлов -> список ключевых блоков по `block` строк (как в Veda)."""
    sub = TILE // block
    blocks = (index.to(torch.int32) * sub)[..., None] + torch.arange(sub, device=index.device, dtype=torch.int32)
    offsets = torch.arange(sub, device=valid_count.device) * block
    valid = (valid_count[:, None] - offsets).clamp_(0, block)
    return (blocks.flatten(-2).contiguous(), (count * sub).contiguous(),
            valid.flatten().to(torch.int32).contiguous())


def attend_int8(q, k, v, index, count, valid_count, softmax_scale, warps=4, stages=3):
    slots, heads, dim = q.shape
    q, k = q.contiguous(), k.contiguous()
    q_int8, q_scale = quantize(q, TILE, pre_scale=softmax_scale * LOG2E)
    k_int8, k_scale = quantize(k, KEY_BLOCK)
    blocks, block_count, valid = key_blocks(index, count, valid_count, KEY_BLOCK)
    v16 = v.to(torch.float16)
    out = torch.empty(q.shape, dtype=torch.float16, device=q.device)
    _attention_int8_kernel[(slots // TILE, heads)](
        q_int8, k_int8, v16, q_scale, k_scale, blocks, block_count, valid, out,
        q.stride(0), q.stride(1), v16.stride(0), v16.stride(1),
        q_scale.stride(0), k_scale.stride(0), blocks.stride(0), blocks.stride(1),
        n_q_tiles=slots // TILE, D=dim, BLK=KEY_BLOCK, BLOCK_M=TILE, num_warps=warps, num_stages=stages)
    return out


@functools.lru_cache(maxsize=None)
def fp16_launch(capability):
    """(BLOCK_M, warps, stages) по архитектуре: у V100 96 KB shared memory — меньше стадий."""
    if capability[0] < 8:
        return 64, 4, 1
    return 64, 4, 2


def attend_fp16(q, k, v, index, count, valid_count, softmax_scale, launch=None):
    slots, heads, dim = q.shape
    q16, k16, v16 = (t.to(torch.float16).contiguous() for t in (q, k, v))
    blocks, block_count, valid = key_blocks(index, count, valid_count, KEY_BLOCK)
    block_m, warps, stages = launch or fp16_launch(torch.cuda.get_device_capability(q.device))
    out = torch.empty(q16.shape, dtype=torch.float16, device=q.device)
    _attention_fp16_kernel[(slots // block_m, heads)](
        q16, k16, v16, blocks, block_count, valid, out,
        q16.stride(0), q16.stride(1), k16.stride(0), k16.stride(1), v16.stride(0), v16.stride(1),
        out.stride(0), out.stride(1), blocks.stride(0), blocks.stride(1), slots // TILE,
        float(softmax_scale * LOG2E), D=dim, BLK=KEY_BLOCK, BLOCK_M=block_m, SUB=TILE // block_m,
        num_warps=warps, num_stages=stages)
    return out
