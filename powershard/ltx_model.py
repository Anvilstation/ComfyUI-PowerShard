"""Worker-side вычисления LTX-2 / 2.3 / 2.5 (LTXAVModel) и LTX-Video (LTXVModel) поверх родных классов comfy.

Родная модель даёт дерево параметров (имена = checkpoint), patchifiers, позиционные частоты
(_precompute_freqs_cis), guide-маску (_build_guide_self_attention_mask), подготовку контекста и
коннекторы текста. Forward блоков, attention и FFN заменяются методами экземпляров (без глобальных
monkey patches и без comfy-kitchen ядер): FP32 residual/модуляция/RMSNorm/RoPE, FP16 GEMM, FP16
attention со статическими степенями двойки для q/k.

Sequence parallel: делятся только ВИДЕО-токены (их десятки тысяч); аудио-токены (сотни)
реплицированы на всех rank.
  * видео self-attention — token (all-gather K/V) или Ulysses (all-to-all heads), как у Wan;
    с guide-маской (IC-LoRA/keyframes, strength != 1) — token-путь с маской по глобальным строкам;
  * текстовая cross-attention и a2v (видео-запросы -> аудио K/V) — локально;
  * v2a (аудио-запросы -> ВСЕ видео-ключи) — частичная softmax по локальным ключам (online,
    FP32) и all-reduce (MAX m, SUM o/l): точное значение полной attention;
  * timestep-модуляция: уникальные значения timestep -> таблица AdaLN, строки берутся индексом
    только для локальных токенов (без [B, T, 9*dim] тензора);
  * выходная голова считается для локальных строк, собирается [B, T, 128].
Каждый FSDP unit вызывается одинаковое число раз и в одном порядке на всех rank.
"""
import math
import types
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from .config import shard_bounds
from .operations import Linear
from .wan_model import (LayerNorm, GroupNorm, Conv1d, Conv2d, Conv3d, Embedding, power2, scaled_half, v_scale,
                        run_attention, sequence_attention, gather_tokens, chunked_mlp, new_sequence_state)


class RMSNorm(nn.Module):
    """RMSNorm с опциональным весом и eps=None (как torch.nn.RMSNorm); вычисление FP32."""

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None, **kwargs):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        y = x.float()
        eps = self.eps if self.eps is not None else torch.finfo(torch.float32).eps
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
        return y if self.weight is None else y * self.weight.float()


class LTXOperations:
    Linear = Linear
    RMSNorm = RMSNorm
    LayerNorm = LayerNorm
    GroupNorm = GroupNorm
    Conv1d = Conv1d
    Conv2d = Conv2d
    Conv3d = Conv3d
    Embedding = Embedding


AUX_UNIT_PATHS = ("patchify_proj", "audio_patchify_proj", "adaln_single", "audio_adaln_single", "prompt_adaln_single",
                  "audio_prompt_adaln_single", "av_ca_video_scale_shift_adaln_single", "av_ca_a2v_gate_adaln_single",
                  "av_ca_audio_scale_shift_adaln_single", "av_ca_v2a_gate_adaln_single", "caption_projection",
                  "audio_caption_projection", "video_embeddings_connector", "audio_embeddings_connector",
                  "proj_out", "audio_proj_out")
# Маленькие GEMM — FP32 (как autocast(float32) у timestep/caption модулей); коннекторы текста тоже FP32.
FP32_LINEAR_PREFIXES = ("adaln_single.", "audio_adaln_single.", "prompt_adaln_single.", "audio_prompt_adaln_single.",
                        "av_ca_", "patchify_proj", "audio_patchify_proj", "proj_out", "audio_proj_out",
                        "caption_projection.", "audio_caption_projection.", "video_embeddings_connector.",
                        "audio_embeddings_connector.")
FP32_PARAMETER_SUFFIXES = ("scale_shift_table", "keyframes_abs_pos_embedding", "learnable_registers")
LTX_KWARGS = ("denoise_mask", "audio_denoise_mask", "guide_attention_entries", "ref_audio", "generated_keyframes",
              "latent_shapes")
UNBATCHED_KWARGS = ("guide_attention_entries", "ref_audio", "generated_keyframes", "latent_shapes", "nag_context")


def build_ltx_network(config, dtype=torch.float16):
    from comfy.ldm.lightricks.av_model import LTXAVModel
    from comfy.ldm.lightricks.model import LTXVModel
    cls = LTXAVModel if config["image_model"] == "ltxav" else LTXVModel
    kwargs = {k: v for k, v in config.items() if k != "disable_unet_model_creation"}
    with torch.device("meta"):
        return cls(**kwargs, dtype=dtype, device="meta", operations=LTXOperations)


def make_fp32_parameters(net):
    for module in net.modules():
        for name, p in list(module._parameters.items()):
            if p is not None and any(token in name for token in FP32_PARAMETER_SUFFIXES):
                setattr(module, name, nn.Parameter(torch.empty(p.shape, dtype=torch.float32, device=p.device),
                                                   requires_grad=False))


def fsdp_unit_modules(net):
    units = list(net.transformer_blocks)
    for path in AUX_UNIT_PATHS:
        module = getattr(net, path, None)
        if isinstance(module, nn.Module) and any(True for _ in module.parameters()):
            units.append(module)
    return units


class LTXEntrypoint(nn.Module):
    """FSDP root: forward генератора и препроцессинг текста коннекторами (host extra_conds)."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, command, args, kwargs):
        if command == "preprocess":
            context, = args
            return self.network.preprocess_text_embeds(context.float(), unprocessed=bool(kwargs.get("unprocessed")))
        if command != "forward":
            raise ValueError(f"Неизвестный вызов LTX: {command}")
        x, timestep, context = args
        chunk = getattr(self.network, "_ps_batch_chunk", 0)
        batch = (x[0] if isinstance(x, (list, tuple)) else x).shape[0]
        if not chunk or batch <= chunk:
            return ltx_forward(self.network, x, timestep, context, **kwargs)
        outputs = []
        for a in range(0, batch, chunk):
            b = min(a + chunk, batch)
            part = lambda v, key=None: batch_slice(v, a, b, batch) if key not in UNBATCHED_KWARGS else v  # noqa: E731
            chunk_kwargs = {k: part(v, k) for k, v in kwargs.items()}
            chunk_kwargs["transformer_options"] = dict(kwargs.get("transformer_options") or {}, _ps_rows=(a, b, batch))
            outputs.append(ltx_forward(self.network, part(x), part(timestep), part(context), **chunk_kwargs))
        if isinstance(outputs[0], (list, tuple)):
            return [torch.cat([o[i] for o in outputs], dim=0) for i in range(len(outputs[0]))]
        return torch.cat(outputs, dim=0)


def batch_slice(value, a, b, batch):
    if isinstance(value, torch.Tensor):
        return value[a:b] if value.ndim and value.shape[0] == batch else value
    if isinstance(value, (list, tuple)):
        return type(value)(batch_slice(v, a, b, batch) for v in value)
    return value


# ------------------------------------------------------------------ numerics
def rms_norm(x):
    x = x.float()
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


def rms_adaln(x, scale, shift):
    return rms_norm(x) * (1 + scale) + shift


def apply_rope(x, pe):
    """LTX RoPE (interleaved или split) в FP32: x [B,L,H*D] -> [B,L,H,D]; pe=(rotation [B,L,H,D/2,2,2], split)."""
    rotation, split = pe
    batch, length = x.shape[0], x.shape[1]
    heads = rotation.shape[2]
    t = x.float().reshape(batch, length, heads, -1)
    dim = t.shape[-1]
    if split:
        t = t.reshape(batch, length, heads, 2, dim // 2).movedim(-2, -1).unsqueeze(-2)
    else:
        t = t.reshape(batch, length, heads, dim // 2, 1, 2)
    out = rotation[..., 0].float() * t[..., 0] + rotation[..., 1].float() * t[..., 1]
    if split:
        out = out.movedim(-1, -2)
    return out.reshape(batch, length, heads, dim)


def half_mask(mask):
    """Аддитивная маска -> FP16 без -inf/переполнения (exp(-1e4) = 0 в FP32 softmax)."""
    return None if mask is None else mask.float().clamp(min=-1e4, max=1e4).half()


def sdpa16(q16, k16, v16, scale, mask=None):
    """BLHD FP16 -> BLHD FP16 через torch SDPA (маски/head_dim вне сертифицированной геометрии)."""
    if q16.shape[1] == 0 or k16.shape[1] == 0:
        return q16.new_zeros(q16.shape[:-1] + (v16.shape[-1],))
    out = F.scaled_dot_product_attention(q16.transpose(1, 2), k16.transpose(1, 2), v16.transpose(1, 2),
                                         attn_mask=half_mask(mask), scale=scale)
    return out.transpose(1, 2)


def sdpa_scaled(q, k, v, sq, sk, mask=None):
    """FP32 q/k/v BLHD -> FP32: q/sq, k/sk (статические), V по своему max — в FP16 без переполнения."""
    sv = v_scale(v, False)
    out = sdpa16(scaled_half(q, sq), scaled_half(k, sk), scaled_half(v, sv), q.shape[-1] ** -.5 * sq * sk, mask)
    return out.float() * sv


def attend(attn, q, k, v, mask=None):
    sq, sk = attn._ps_qk_scales[:2]
    if mask is None and attn._ps_dispatch:
        return run_attention(attn, q, k, v, sq, sk, attn._ps_group)
    return sdpa_scaled(q, k, v, sq, sk, mask)


def guide_mask_attention(q16, k16, v16, scale, guide, row0):
    """GuideAttentionMask по глобальным строкам запросов [row0, row0+n): noisy / tracked / прочие."""
    n = q16.shape[1]
    out = q16.new_empty(q16.shape[:-1] + (v16.shape[-1],))
    start, end = guide.guide_start, guide.guide_start + guide.tracked_count
    for lo, hi, kind in ((0, start, "noisy"), (start, end, "tracked"), (end, row0 + n, None)):
        a, b = max(lo, row0) - row0, min(hi, row0 + n) - row0
        if a >= b:
            continue
        if kind == "noisy":
            mask = guide.noisy_mask
        elif kind == "tracked":
            mask = guide.tracked_mask[:, :, row0 + a - start:row0 + b - start]
        else:
            mask = None
        out[:, a:b] = sdpa16(q16[:, a:b], k16, v16, scale, mask)
    return out


def partial_attention(q, k, v, scale, chunk, distributed):
    """softmax(q k^T) v по ключам, распределённым между rank (online softmax, FP32, all-reduce)."""
    qh = q.float().transpose(1, 2)                                   # [B,H,Lq,D]
    batch, heads, length, _ = qh.shape
    m = qh.new_full((batch, heads, length, 1), float("-inf"))
    denom = qh.new_zeros((batch, heads, length, 1))
    acc = qh.new_zeros((batch, heads, length, v.shape[-1]))
    for start in range(0, k.shape[1], chunk):
        kc = k[:, start:start + chunk].float().transpose(1, 2)
        vc = v[:, start:start + chunk].float().transpose(1, 2)
        s = torch.matmul(qh, kc.transpose(-1, -2)) * scale
        new_m = torch.maximum(m, s.amax(-1, keepdim=True))
        p = torch.exp(s - new_m)
        correction = torch.exp(m - new_m)
        denom = denom * correction + p.sum(-1, keepdim=True)
        acc = acc * correction + torch.matmul(p, vc)
        m = new_m
        del s, p, kc, vc
    if distributed:
        top = m.clone()
        dist.all_reduce(top, op=dist.ReduceOp.MAX)
        factor = torch.exp(m - top)                                 # -inf (нет ключей) -> 0
        acc, denom = acc * factor, denom * factor
        dist.all_reduce(acc)
        dist.all_reduce(denom)
    return (acc / denom).transpose(1, 2)                            # [B,Lq,H,D]


def apply_gate(attn, x, out):
    if getattr(attn, "to_gate_logits", None) is None:
        return out
    batch, length = out.shape[:2]
    gates = 2.0 * torch.sigmoid(attn.to_gate_logits(x).float())   # [B,L,H]
    return (out.view(batch, length, attn.heads, -1) * gates.unsqueeze(-1)).view(batch, length, -1)


# ------------------------------------------------------------ attention forms
def generic_attention_forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    """Замена CrossAttention.forward для вызовов вне блоков (коннекторы текста и т.п.), локально."""
    from comfy.ldm.lightricks.model import GuideAttentionMask
    stg = context is None and bool((transformer_options or {}).get("stg_skip_self_attn", False))
    ctx = x if context is None else context
    batch, n = x.shape[0], x.shape[1]
    heads, dim = self.heads, self.dim_head
    if stg:
        out = self.to_v(ctx).float()
    else:
        q, k = self.q_norm(self.to_q(x)), self.k_norm(self.to_k(ctx))
        if pe is not None:
            q, k = apply_rope(q, pe), apply_rope(k, pe if k_pe is None else k_pe)
        else:
            q, k = q.float().view(batch, n, heads, dim), k.float().view(batch, -1, heads, dim)
        v = self.to_v(ctx).float().view(batch, -1, heads, dim)
        if isinstance(mask, GuideAttentionMask):
            sq, sk = self._ps_qk_scales[:2]
            sv = v_scale(v, False)
            out = guide_mask_attention(scaled_half(q, sq), scaled_half(k, sk), scaled_half(v, sv),
                                       dim ** -.5 * sq * sk, mask, 0).float() * sv
        else:
            out = attend(self, q, k, v, mask)
        out = out.reshape(batch, n, heads * dim)
    return self.to_out[0](apply_gate(self, x, out)).float()


def video_self_attention(attn, x, run, stg):
    """Видео attn1: распределённая по токенам (Wan sequence_attention) или с guide-маской."""
    batch, n, _ = x.shape
    heads, dim = attn.heads, attn.dim_head
    if stg:  # STG: self-attention -> V (локально, без обмена; флаг одинаков на всех rank)
        out = attn.to_v(x).float()
    else:
        q = apply_rope(attn.q_norm(attn.to_q(x)), run.v_pe)
        k = apply_rope(attn.k_norm(attn.to_k(x)), run.v_pe)
        v = attn.to_v(x).float().view(batch, n, heads, dim)
        if run.guide_mask is not None:
            out = guided_self_attention(attn, q, k, v, run)
        else:
            out = sequence_attention(attn, q, k, v)
        out = out.reshape(batch, n, heads * dim)
    return attn.to_out[0](apply_gate(attn, x, out)).float()


def guided_self_attention(attn, q, k, v, run):
    state = attn._ps_sequence
    sq, sk = attn._ps_qk_scales[:2]
    distributed = state["enabled"]
    sv = v_scale(v, distributed)
    k16, v16 = scaled_half(k, sk), scaled_half(v, sv)
    if distributed:
        state["scale_collectives"] += 1
        kv = gather_tokens(torch.stack((k16, v16), dim=2), state["total"])
        state["kv_collectives"] += 1
        state["kv_gathered_bytes"] += kv.numel() * kv.element_size()
        k16, v16 = kv.unbind(2)
    out = guide_mask_attention(scaled_half(q, sq), k16, v16, q.shape[-1] ** -.5 * sq * sk, run.guide_mask, run.row0)
    attn._ps_attention.counts["ltx_guide_mask:sdpa"] += 1
    return out.float() * sv


def local_self_attention(attn, x, pe, stg):
    """Аудио attn1 (реплицированные токены): локально, SDPA (head_dim 64)."""
    batch, n, _ = x.shape
    heads, dim = attn.heads, attn.dim_head
    if stg:
        out = attn.to_v(x).float()
    else:
        q, k = apply_rope(attn.q_norm(attn.to_q(x)), pe), apply_rope(attn.k_norm(attn.to_k(x)), pe)
        v = attn.to_v(x).float().view(batch, n, heads, dim)
        out = attend(attn, q, k, v).reshape(batch, n, heads * dim)
    return attn.to_out[0](apply_gate(attn, x, out)).float()


def text_attention(attn, x, context, mask, nag=None, negative=None):
    """Cross-attention к тексту для локальных строк; NAG — для строк положительной ветки."""
    from .accel import nag_combine
    batch, n, _ = x.shape
    heads, dim = attn.heads, attn.dim_head
    q = attn.q_norm(attn.to_q(x)).float().view(batch, n, heads, dim)
    k = attn.k_norm(attn.to_k(context)).float().view(context.shape[0], -1, heads, dim)
    v = attn.to_v(context).float().view(context.shape[0], -1, heads, dim)
    out = attend(attn, q, k, v, mask)
    if nag is not None and negative is not None and nag["rows"]:
        rows = torch.tensor(nag["rows"], device=x.device, dtype=torch.long)
        kn = attn.k_norm(attn.to_k(negative)).float().view(negative.shape[0], -1, heads, dim)
        vn = attn.to_v(negative).float().view(negative.shape[0], -1, heads, dim)
        neg = attend(attn, q.index_select(0, rows), kn, vn)
        mixed = nag_combine(out.index_select(0, rows).reshape(len(nag["rows"]), n, -1), neg.reshape(len(nag["rows"]), n, -1),
                            nag["scale"], nag["tau"], nag["alpha"])
        out = out.index_copy(0, rows, mixed.view(len(nag["rows"]), n, heads, dim))
    out = out.reshape(batch, n, heads * dim)
    return attn.to_out[0](apply_gate(attn, x, out)).float()


def a2v_attention(attn, vx, ax, run):
    """Видео-запросы (локальные) -> аудио K/V (все): локально."""
    batch, n, _ = vx.shape
    heads, dim = attn.heads, attn.dim_head
    q = apply_rope(attn.q_norm(attn.to_q(vx)), run.v_cross_pe)
    k = apply_rope(attn.k_norm(attn.to_k(ax)), run.a_cross_pe)
    v = attn.to_v(ax).float().view(batch, -1, heads, dim)
    out = attend(attn, q, k, v).reshape(batch, n, heads * dim)
    return attn.to_out[0](apply_gate(attn, vx, out)).float()


def v2a_attention(attn, ax, vx, run):
    """Аудио-запросы -> все видео-ключи: частичная attention по локальным ключам + all-reduce."""
    batch, length, _ = ax.shape
    heads, dim = attn.heads, attn.dim_head
    state = attn._ps_sequence
    q = apply_rope(attn.q_norm(attn.to_q(ax)), run.a_cross_pe)
    k = apply_rope(attn.k_norm(attn.to_k(vx)), run.v_cross_pe)
    v = attn.to_v(vx).float().view(batch, vx.shape[1], heads, dim)
    out = partial_attention(q, k, v, dim ** -.5, run.attention_chunk, state["enabled"])
    if state["enabled"]:
        state["v2a_collectives"] = state.get("v2a_collectives", 0) + 3
    attn._ps_attention.counts["ltx_v2a_partial:fp32"] += 1
    out = out.reshape(batch, length, heads * dim)
    return attn.to_out[0](apply_gate(attn, ax, out)).float()


def ltx_ff_forward(self, x):
    return chunked_mlp(self, x, self.net[0].proj, self.net[2], "LTX_FFN_chunks")


# ------------------------------------------------------------------ timestep
class IndexedTimestep:
    """AdaLN по уникальным значениям timestep: table [U, P*dim], index [B, N] (N=1 — broadcast) или None."""

    def __init__(self, table, index):
        self.table, self.index = table, index

    def local(self, a, b):
        if self.index is None or self.index.shape[1] == 1:
            return self
        return IndexedTimestep(self.table, self.index[:, a:b])

    def values(self, table_rows, indices=slice(None)):
        params = table_rows.shape[0]
        tab = self.table.view(self.table.shape[0], params, -1)[:, indices] + table_rows[indices].float().unsqueeze(0)
        if self.index is None or tab.shape[0] == 1:
            return tuple(tab[0, j].view(1, 1, -1) for j in range(tab.shape[1]))
        return tab[self.index].unbind(2)                               # [B, n, dim] каждый

    def rows(self):
        if self.index is None or self.table.shape[0] == 1:
            return self.table[:1].view(1, 1, -1)
        return self.table[self.index]


def plain_values(table_rows, timestep, indices=slice(None)):
    """Native get_ada_values для тензора timestep [B|1, N, P*dim] (аудио, реплицированное)."""
    params = table_rows.shape[0]
    value = timestep.reshape(timestep.shape[0], timestep.shape[1], params, -1)[:, :, indices]
    return (table_rows[indices].float()[None, None] + value).unbind(2)


ADALN_ARGS = {"resolution": None, "aspect_ratio": None}


def indexed_adaln(module, rows):
    values, inverse = torch.unique(rows.reshape(-1).float(), sorted=True, return_inverse=True)
    modulation, embedded = module(values, ADALN_ARGS, batch_size=values.shape[0], hidden_dtype=torch.float32)
    index = inverse.reshape(rows.shape)
    return IndexedTimestep(modulation.float(), index), IndexedTimestep(embedded.float(), index)


# --------------------------------------------------------------------- blocks
def text_cross(block, x, context, attn, table, prompt_table, timestep, prompt_timestep, mask, nag, negative,
               indexed, prenorm=True):
    """_apply_text_cross_attention / apply_cross_attention_adaln (LTX 2.3+) с NAG."""
    if block.cross_attention_adaln:
        if indexed:
            shift_q, scale_q, gate = timestep.values(table, slice(6, 9))
        else:
            shift_q, scale_q, gate = plain_values(table, timestep, slice(6, 9))
        batch = x.shape[0]
        shift_kv, scale_kv = (prompt_table.float()[None, None]
                              + prompt_timestep.float().reshape(batch, prompt_timestep.shape[1], 2, -1)).unbind(2)
        neg = None
        if negative is not None and nag is not None and nag["rows"]:
            rows = torch.tensor(nag["rows"], device=x.device, dtype=torch.long)
            neg = (negative.float().expand(len(nag["rows"]), -1, -1) * (1 + scale_kv.index_select(0, rows))
                   + shift_kv.index_select(0, rows))
        return text_attention(attn, rms_adaln(x, scale_q, shift_q), context.float() * (1 + scale_kv) + shift_kv,
                              mask, nag, neg) * gate
    neg = None
    if negative is not None and nag is not None and nag["rows"]:
        neg = negative.float().expand(len(nag["rows"]), -1, -1)
    return text_attention(attn, rms_norm(x) if prenorm else x.float(), context, mask, nag, neg)


def av_block_forward(self, vx, ax, run, stg=False):
    """BasicAVTransformerBlock: vx — локальные видео-строки [B,n,dim] FP32, ax — все аудио-токены."""
    flags = run.flags
    run_vx = flags.get("run_vx", True)
    run_ax = flags.get("run_ax", True) and ax.numel() > 0
    run_a2v = run_vx and flags.get("a2v_cross_attn", True) and ax.numel() > 0
    run_v2a = run_ax and flags.get("v2a_cross_attn", True)
    table, audio_table = self.scale_shift_table, self.audio_scale_shift_table
    if run_vx:
        shift, scale = run.V.values(table, slice(0, 2))
        y = video_self_attention(self.attn1, rms_adaln(vx, scale, shift), run, stg)
        vx = vx + y * run.V.values(table, slice(2, 3))[0]
        vx = vx + text_cross(self, vx, run.v_ctx, self.attn2, table, getattr(self, "prompt_scale_shift_table", None),
                             run.V, run.v_prompt, run.attention_mask, run.nag, run.v_neg, True)
    if run_ax:
        shift, scale = plain_values(audio_table, run.A, slice(0, 2))
        y = local_self_attention(self.audio_attn1, rms_norm(ax) * (1 + scale) + shift, run.a_pe, stg)
        ax = ax + y * plain_values(audio_table, run.A, slice(2, 3))[0]
        ax = ax + text_cross(self, ax, run.a_ctx, self.audio_attn2, audio_table,
                             getattr(self, "audio_prompt_scale_shift_table", None), run.A, run.a_prompt,
                             run.attention_mask, run.nag if run.nag and run.nag.get("audio") else None,
                             run.a_neg, False)
    if run_a2v or run_v2a:
        ax_norm = rms_norm(ax)
        if run_a2v:
            scale_a, shift_a = plain_values(self.scale_shift_table_a2v_ca_audio[:4], run.a_ca)[:2]
            scale_v, shift_v = run.V_ca.values(self.scale_shift_table_a2v_ca_video[:4], slice(0, 2))
            out = a2v_attention(self.audio_to_video_attn, rms_adaln(vx, scale_v, shift_v),
                                ax_norm * (1 + scale_a) + shift_a, run)
            vx = vx + out * run.V_gate.values(self.scale_shift_table_a2v_ca_video[4:])[0]
        if run_v2a:
            scale_a, shift_a = plain_values(self.scale_shift_table_a2v_ca_audio[:4], run.a_ca)[2:4]
            scale_v, shift_v = run.V_ca.values(self.scale_shift_table_a2v_ca_video[:4], slice(2, 4))
            out = v2a_attention(self.video_to_audio_attn, ax_norm * (1 + scale_a) + shift_a,
                                rms_adaln(vx, scale_v, shift_v), run)
            ax = ax + out * plain_values(self.scale_shift_table_a2v_ca_audio[4:], run.v2a_gate)[0]
    if run_vx:
        shift, scale = run.V.values(table, slice(3, 5))
        vx = vx + self.ff(rms_adaln(vx, scale, shift)).float() * run.V.values(table, slice(5, 6))[0]
    if run_ax:
        shift, scale = plain_values(audio_table, run.A, slice(3, 5))
        ax = ax + self.audio_ff(rms_norm(ax) * (1 + scale) + shift).float() * plain_values(audio_table, run.A, slice(5, 6))[0]
    tracker = getattr(self, "_ps_tracker", None)
    if tracker is not None:
        tracker.observe(self._ps_finite_slot, vx)
    return vx, ax


def v_block_forward(self, vx, ax, run, stg=False):
    """BasicTransformerBlock (LTX-Video без аудио)."""
    table = self.scale_shift_table
    shift, scale, gate = run.V.values(table, slice(0, 3))
    vx = vx + video_self_attention(self.attn1, rms_adaln(vx, scale, shift), run, stg) * gate
    vx = vx + text_cross(self, vx, run.v_ctx, self.attn2, table, getattr(self, "prompt_scale_shift_table", None),
                         run.V, run.v_prompt, run.attention_mask, run.nag, run.v_neg, True, prenorm=False)
    shift, scale, gate = run.V.values(table, slice(3, 6))
    y = rms_norm(vx)
    vx = vx + self.ff(y * (1 + scale) + shift).float() * gate
    tracker = getattr(self, "_ps_tracker", None)
    if tracker is not None:
        tracker.observe(self._ps_finite_slot, vx)
    return vx, ax


# --------------------------------------------------------------- installation
def attention_modules(net):
    from comfy.ldm.lightricks.model import CrossAttention
    return {name: m for name, m in net.named_modules() if isinstance(m, CrossAttention)}


def uses_rope(name):
    leaf = name.rsplit(".", 1)[-1]
    return leaf in ("attn1", "audio_attn1", "audio_to_video_attn", "video_to_audio_attn")


def configure_ltx(net, config, options, dispatcher, distributed_config):
    """На meta-модели до FSDP: FP16/FP32 политика Linear, forwards, состояние sequence parallel.

    config — геометрия checkpoint (dict), distributed_config — DistributedConfig (sequence_mode/comm dtype).
    """
    from comfy.ldm.lightricks.model import FeedForward
    from .fp16_safe import FiniteTracker, install_safe_operations
    tracker = FiniteTracker(options.debug_finite)
    install_safe_operations(net, tracker, options.fp16_safe)
    for name, module in net.named_modules():
        if name.startswith(FP32_LINEAR_PREFIXES) and isinstance(module, Linear):
            module._ps_safe, module._ps_fp32 = True, True
    state = new_sequence_state(distributed_config)
    net._ps_sequence = state
    geometry = (dispatcher.policy or {}).get("geometry") or {}
    heads, head_dim = int(net.num_attention_heads), int(net.attention_head_dim)
    for name, module in attention_modules(net).items():
        module.forward = types.MethodType(generic_attention_forward, module)
        module.num_heads, module.head_dim = module.heads, module.dim_head
        module._ps_attention = dispatcher
        module._ps_safe = options.fp16_safe
        module._ps_sequence = state
        leaf = name.rsplit(".", 1)[-1]
        module._ps_group = "ltx_" + ("connector" if "connector" in name else leaf)
        module._ps_qk_scales = (1., 1., 1.)
        # Сертифицированный kernel — только видео-геометрия (heads x head_dim как у probes).
        module._ps_dispatch = (module.heads == heads and module.dim_head == head_dim
                               and geometry.get("head_dim", head_dim) == head_dim)
    for name, module in net.named_modules():
        if isinstance(module, FeedForward):
            module.forward = types.MethodType(ltx_ff_forward, module)
            module._ps_mlp_policy = options
            module._ps_label = name
    av = config["image_model"] == "ltxav"
    for index, block in enumerate(net.transformer_blocks):
        block.forward = types.MethodType(av_block_forward if av else v_block_forward, block)
        block._ps_block_index = index
    net._ps_av = av
    net._ps_batch_chunk = options.batch_chunk
    net._ps_attention_chunk = options.attention_chunk
    return tracker, state


def qk_norm_names(net):
    names = {}
    for name, module in attention_modules(net).items():
        for norm in ("q_norm", "k_norm"):
            sub = getattr(module, norm, None)
            if sub is not None and getattr(sub, "weight", None) is not None:
                names[f"{name}.{norm}.weight"] = (name, norm)
    return names


def apply_qk_scales(net, maxima):
    """RMSNorm по всему inner_dim: |y_j| <= sqrt(n)*max|w|; RoPE даёт sqrt(2). Цель |q/s| <= 128."""
    for name, module in attention_modules(net).items():
        values = []
        for norm in ("q_norm", "k_norm"):
            sub = getattr(module, norm)
            maximum = maxima.get(f"{name}.{norm}.weight", 1.)
            if not math.isfinite(maximum):
                raise FloatingPointError(f"Non-finite {name}.{norm} weights")
            width = sub.weight.shape[0] if getattr(sub, "weight", None) is not None else module.heads * module.dim_head
            bound = (math.sqrt(2.) if uses_rope(name) else 1.) * math.sqrt(width) * maximum * 1.01
            values.append(power2(bound / 128.))
        module._ps_qk_scales = (values[0], values[1], 1.)


# -------------------------------------------------------------------- forward
def positional(net, v_coords, a_coords, frame_rate, av):
    """Частоты RoPE только для локальных видео-строк (и всех аудио), как native _prepare_positional_embeddings."""
    fp32 = torch.float32
    rate = float(frame_rate.reshape(-1)[0]) if isinstance(frame_rate, torch.Tensor) else float(frame_rate)
    coords = v_coords.to(fp32).clone()
    coords[:, 0] = coords[:, 0] * (1.0 / rate)
    v_pe = net._precompute_freqs_cis(coords, dim=net.inner_dim, out_dtype=fp32, max_pos=net.positional_embedding_max_pos,
                                     use_middle_indices_grid=net.use_middle_indices_grid,
                                     num_attention_heads=net.num_attention_heads)
    if not av:
        return v_pe, None, None, None
    a_pe = net._precompute_freqs_cis(a_coords, dim=net.audio_inner_dim, out_dtype=fp32,
                                     max_pos=net.audio_positional_embedding_max_pos,
                                     use_middle_indices_grid=net.use_middle_indices_grid,
                                     num_attention_heads=net.audio_num_attention_heads)
    max_pos = max(net.positional_embedding_max_pos[0], net.audio_positional_embedding_max_pos[0])
    v_cross = net._precompute_freqs_cis(coords[:, 0:1, :], dim=net.audio_cross_attention_dim, out_dtype=fp32,
                                        max_pos=[max_pos], use_middle_indices_grid=True,
                                        num_attention_heads=net.audio_num_attention_heads)
    a_cross = net._precompute_freqs_cis(a_coords[:, 0:1, :], dim=net.audio_cross_attention_dim, out_dtype=fp32,
                                        max_pos=[max_pos], use_middle_indices_grid=True,
                                        num_attention_heads=net.audio_num_attention_heads)
    return v_pe, v_cross, a_pe, a_cross


def resolve_guides(guide_entries, kf_grid_mask):
    """Как native LTXVModel._process_input: число выживших токенов каждого guide после grid mask."""
    total = sum(e["pre_filter_count"] for e in guide_entries)
    if total != len(kf_grid_mask):
        raise ValueError(f"guide pre_filter_counts ({total}) != keyframe grid mask length ({len(kf_grid_mask)})")
    resolved, offset = [], 0
    for entry in guide_entries:
        count = entry["pre_filter_count"]
        resolved.append({**entry, "surviving_count": int(kf_grid_mask[offset:offset + count].sum().item())})
        offset += count
    return resolved


def ltx_forward(net, x, timestep, context, attention_mask=None, frame_rate=25, transformer_options=None,
                keyframe_idxs=None, denoise_mask=None, _powershard_flags=None, _powershard_block_cache=None,
                _powershard_nag=None, nag_context=None, **kwargs):
    """Эквивалент LTXAVModel/LTXVModel._forward с разбиением видео-токенов по rank."""
    import comfy.ldm.lightricks.model as ltx
    from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
    unknown = sorted(set(k for k, v in kwargs.items() if v is not None) - set(LTX_KWARGS))
    if unknown:
        raise ValueError(f"PowerShard LTX: conditioning {unknown} не поддерживается")
    options = dict(transformer_options or {})
    flags = dict(_powershard_flags or {})
    av = net._ps_av
    state = net._ps_sequence
    fp32 = torch.float32
    if isinstance(timestep, (tuple, list)) and len(timestep) == 2:
        v_t, a_t = timestep
    else:
        v_t = a_t = timestep
    parts = list(x) if isinstance(x, (list, tuple)) else [x]
    vx_lat = parts[0].float()
    batch = vx_lat.shape[0]
    # ---- вход (как LTXVModel._process_input, но проекция только локальных строк)
    if av:
        ax_lat = parts[1].float() if len(parts) > 1 else vx_lat.new_zeros(
            (batch, net.num_audio_channels, 0, net.audio_frequency_bins))
    orig_shape = list(vx_lat.shape)
    tokens, latent_coords = net.patchifier.patchify(vx_lat)
    pixel_coords = latent_to_pixel_coords(latent_coords=latent_coords, scale_factors=net.vae_scale_factors,
                                          causal_fix=net.causal_temporal_positioning)
    grid_mask, num_guide, resolved, patched_shape = None, 0, None, None
    if keyframe_idxs is not None and keyframe_idxs.shape[2] > 0:
        per_frame = net.tokens_per_latent_frame(orig_shape)
        if keyframe_idxs.shape[2] % per_frame != 0:
            raise ValueError(f"keyframe_idxs holds {keyframe_idxs.shape[2]} tokens, not a whole number of "
                             f"{per_frame}-token latent frames (crop guides / separate generated keyframes before upscaling)")
        if denoise_mask is None:
            raise ValueError("LTX keyframes/guides требуют denoise_mask")
        patched_shape = list(tokens.shape)
        mask_tokens = net.patchifier.patchify(denoise_mask.float())[0]
        grid_mask = ~torch.any(mask_tokens < 0, dim=-1)[0]
        tokens = tokens[:, grid_mask, :]
        pixel_coords = pixel_coords[:, :, grid_mask, ...]
        kf_grid_mask = grid_mask[-keyframe_idxs.shape[2]:]
        if kwargs.get("guide_attention_entries"):
            resolved = resolve_guides(kwargs["guide_attention_entries"], kf_grid_mask)
        keyframe_idxs = keyframe_idxs[..., kf_grid_mask, :]
        if keyframe_idxs.shape[2] > 0:
            pixel_coords[:, :, -keyframe_idxs.shape[2]:, :] = keyframe_idxs.to(pixel_coords.dtype)
        num_guide = keyframe_idxs.shape[2]
    total = tokens.shape[1]
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    last_a, last_b = shard_bounds(total, world - 1, world)
    state.update(total=total, enabled=world > 1 and last_a < last_b, kv_collectives=0, kv_gathered_bytes=0,
                 out_collectives=0, out_exchanged_bytes=0, scale_collectives=0, head_gathers=0, v2a_collectives=0,
                 guide_mask_tokens=0, blocks_skipped=False)
    a, b = shard_bounds(total, rank, world) if state["enabled"] else (0, total)
    vx = net.patchify_proj(tokens[:, a:b].contiguous()).float()
    if net.keyframes_abs_pos_embedding is not None:
        marker = net.keyframes_abs_pos_mask(pixel_coords, orig_shape, grid_mask, num_guide, kwargs.get("generated_keyframes"))
        vx = vx + marker[:, a:b].unsqueeze(-1).float() * net.keyframes_abs_pos_embedding.float()
    del tokens
    ax, a_coords, ref_len = None, None, 0
    if av:
        ax, a_coords = net.a_patchifier.patchify(ax_lat)
        ref_audio = kwargs.get("ref_audio")
        if ref_audio is not None:
            ref_tokens = ref_audio["tokens"].to(dtype=ax.dtype, device=ax.device)
            if ref_tokens.shape[0] < ax.shape[0]:
                ref_tokens = ref_tokens.expand(ax.shape[0], -1, -1)
            ref_len = ref_tokens.shape[1]
            p = net.a_patchifier
            per_latent = p.hop_length * p.audio_latent_downsample_factor / p.sample_rate
            ref_start = p._get_audio_latent_time_in_sec(0, ref_len, fp32, ax.device)
            ref_end = p._get_audio_latent_time_in_sec(1, ref_len + 1, fp32, ax.device)
            offset = ref_end[-1].item() + per_latent
            ref_start = (ref_start - offset).unsqueeze(0).expand(batch, -1).unsqueeze(1)
            ref_end = (ref_end - offset).unsqueeze(0).expand(batch, -1).unsqueeze(1)
            target_len = ax.shape[1]
            ax = torch.cat([ref_tokens, ax], dim=1)
            a_coords = torch.cat([torch.stack([ref_start, ref_end], dim=-1).to(a_coords), a_coords], dim=2)
        ax = net.audio_patchify_proj(ax).float()
    # ---- timestep: уникальные значения -> AdaLN, индекс строк
    ts = v_t.float()
    if grid_mask is not None and ts.ndim > 1:
        ts = ts[:, grid_mask]
    scaled = ts * net.timestep_scale_multiplier
    rows = scaled.reshape(batch, -1)
    if rows.shape[1] not in (1, total):
        raise ValueError(f"LTX timestep: {rows.shape[1]} значений на {total} видео-токенов")
    V, V_emb = indexed_adaln(net.adaln_single, rows)
    v_prompt = ltx.compute_prompt_timestep(net.prompt_adaln_single, scaled, batch, fp32)
    run = types.SimpleNamespace(flags=flags, V=V.local(a, b), row0=a, attention_chunk=net._ps_attention_chunk,
                                v_prompt=v_prompt, nag=None, v_neg=None, a_neg=None)
    if av:
        at = a_t.float()
        if ref_len > 0:
            if at.dim() <= 1:
                at = at.view(-1, 1).expand(batch, target_len)
            at = torch.cat([at.new_zeros((batch, ref_len) + tuple(at.shape[2:])), at], dim=1)
        a_scaled = at * net.timestep_scale_multiplier
        a_flat = a_scaled.flatten()
        factor = net.av_ca_timestep_scale_multiplier / net.timestep_scale_multiplier
        # max() — по батчу этого forward (при batch_chunk — по части; у sampler-ов ComfyUI sigma общая для всех строк).
        a_ca, _ = net.av_ca_audio_scale_shift_adaln_single(a_flat, ADALN_ARGS, batch_size=batch, hidden_dtype=fp32)
        run.a_ca = a_ca.float().view(batch, -1, a_ca.shape[-1])
        V_ca, _ = indexed_adaln(net.av_ca_video_scale_shift_adaln_single, rows)
        run.V_ca = V_ca.local(a, b)
        gate_a2v, _ = net.av_ca_a2v_gate_adaln_single((a_scaled.max() * factor).reshape(1), ADALN_ARGS,
                                                      batch_size=1, hidden_dtype=fp32)
        run.V_gate = IndexedTimestep(gate_a2v.float(), None)
        gate_v2a, _ = net.av_ca_v2a_gate_adaln_single((scaled.max() * factor).reshape(1), ADALN_ARGS,
                                                      batch_size=1, hidden_dtype=fp32)
        run.v2a_gate = gate_v2a.float().view(1, 1, -1)
        a_mod, a_emb = net.audio_adaln_single(a_flat, ADALN_ARGS, batch_size=batch, hidden_dtype=fp32)
        run.A = a_mod.float().view(batch, -1, a_mod.shape[-1])
        a_emb = a_emb.float().view(batch, -1, a_emb.shape[-1])
        run.a_prompt = ltx.compute_prompt_timestep(net.audio_prompt_adaln_single, a_scaled, batch, fp32)
    # ---- контекст, маски, позиции
    contexts, mask = net._prepare_context(context.float(), batch, [vx, ax] if av else vx, attention_mask)
    run.attention_mask = net._prepare_attention_mask(mask, fp32)
    if av:
        run.v_ctx, run.a_ctx = contexts[0].float(), contexts[1].float()
    else:
        run.v_ctx = contexts.float()
    if _powershard_nag and nag_context is not None:
        negatives, _ = net._prepare_context(nag_context.float(), nag_context.shape[0], [vx, ax] if av else vx, None)
        from .accel import nag_rows
        from .wan_model import batch_rows, batch_total
        run.nag = dict(_powershard_nag, rows=batch_rows(nag_rows(options.get("cond_or_uncond"),
                                                                 batch_total(options, batch)), options))
        if av:
            run.v_neg, run.a_neg = negatives[0].float(), negatives[1].float()
        else:
            run.v_neg = negatives.float()
    run.v_pe, run.v_cross_pe, run.a_pe, run.a_cross_pe = positional(net, pixel_coords[:, :, a:b], a_coords, frame_rate, av)
    merged = {**options, **kwargs, "num_guide_tokens": num_guide}
    if resolved is not None:
        merged["resolved_guide_entries"] = resolved
    probe = vx.new_empty((batch, total, 0))
    run.guide_mask = net._build_guide_self_attention_mask([probe, ax] if av else probe, options, merged)
    if run.guide_mask is not None:
        state["guide_mask_tokens"] = int(run.guide_mask.tracked_count)
    # ---- блоки (+ PowerShard Block Cache)
    # STG читает только LTXAVModel (native LTXVModel флаг игнорирует).
    stg = set(int(i) for i in flags.get("stg_blocks", ())) if av else set()
    blocks = list(net.transformer_blocks)
    from .wan_model import block_cache_spec
    cache = getattr(net, "_ps_block_cache", None)
    spec = block_cache_spec(_powershard_block_cache, options, cache, len(blocks))
    if spec is not None and hasattr(blocks[0], "set_modules_to_forward_prefetch"):
        blocks[0].set_modules_to_forward_prefetch([])  # блок 1 может не понадобиться
    if ax is None:
        ax = vx.new_zeros((batch, 0, 1))
    v0, a0 = vx, ax
    vx, ax = blocks[0](vx, ax, run, 0 in stg)
    skipped = False
    if spec is not None:
        r1 = [vx - v0, ax - a0]
        weights = [1.0, 1.0 / world if state["enabled"] else 1.0]
        skip, entry = cache.decide(spec, r1, weights)
        if skip:
            residual = cache.skip(entry)
            vx, ax = vx + residual[0], ax + residual[1]
            skipped = True
    if not skipped:
        v1, a1 = vx, ax
        for index in range(1, len(blocks)):
            vx, ax = blocks[index](vx, ax, run, index in stg)
        if spec is not None:
            cache.store(spec, r1, [vx - v1, ax - a1])
    state["blocks_skipped"] = skipped
    # ---- выход: голова для локальных строк, сборка [B,T,C]
    emb = V_emb.local(a, b).rows()
    modulation = net.scale_shift_table.float()[None, None] + emb[:, :, None]
    shift, scale = modulation[:, :, 0], modulation[:, :, 1]
    out = net.proj_out(net.norm_out(vx) * (1 + scale) + shift).float()
    if state["enabled"]:
        out = gather_tokens(out, total)
        state["head_gathers"] += 1
    if patched_shape is not None:
        full = out.new_zeros(patched_shape)
        full[:, grid_mask, :] = out
        out = full
    video = net.patchifier.unpatchify(latents=out, output_height=orig_shape[3], output_width=orig_shape[4],
                                      output_num_frames=orig_shape[2],
                                      out_channels=orig_shape[1] // math.prod(net.patchifier.patch_size))
    if not av:
        return video
    if ref_len > 0:
        ax = ax[:, ref_len:]
        if a_emb.shape[1] > 1:
            a_emb = a_emb[:, ref_len:]
    modulation = net.audio_scale_shift_table.float()[None, None] + a_emb[:, :, None]
    a_shift, a_scale = modulation[:, :, 0], modulation[:, :, 1]
    audio = net.audio_proj_out(net.audio_norm_out(ax) * (1 + a_scale) + a_shift).float()
    audio = net.a_patchifier.unpatchify(audio, channels=net.num_audio_channels, freq=net.audio_frequency_bins)
    return net.recombine_audio_and_video_latents(video, audio)
