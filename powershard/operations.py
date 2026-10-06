"""Обычные tensors, FP32 norm и переносимый INT8 ConvRot без tensor subclasses."""
from functools import lru_cache
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


def checkpoint_row_bounds(piece, tile_rows=256):
    """Bounds of stored half weights, using bounded CPU tiles during loading."""
    if piece.device.type != "cpu" or piece.ndim != 2:
        raise ValueError("Weight bounds require a two-dimensional CPU checkpoint shard")
    maximum, row_sum = 0., 0.
    for start in range(0, piece.shape[0], tile_rows):
        values = piece[start:start+tile_rows].to(torch.float16).float().abs()
        maximum = max(maximum, float(values.max()))
        row_sum = max(row_sum, float(values.sum(-1).max()))
    return maximum, row_sum


def matmul_constants(maximum, row_sum, in_features):
    import math
    sb = 2.**max(0, math.ceil(math.log2(max(1., maximum/16384.))))
    # Half rounding/underflow cannot invalidate the absolute-sum bound.
    bound = row_sum/sb*1.001+in_features*2.**-24
    return sb, bound


class Linear(nn.Linear):
    def reset_parameters(self):
        pass  # Только checkpoint loader; создание исключительно на meta.

    def forward(self, x):
        prepared = getattr(self, "_ps_prepared", None)
        if getattr(self, "_ps_safe", False):
            if self._ps_fp32:
                out = F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())
            else:
                from .fp16_safe import safe_linear
                if prepared is None:
                    out = safe_linear(x, self.weight, self.bias, getattr(self, "_ps_weight_bounds", None))
                else:
                    from .fp16_safe import prepared_matmul
                    out = prepared_matmul(x, prepared)
                    if self.bias is not None:
                        out = out + self.bias.float()
        else:
            out = F.linear(x.to(self.weight.dtype), self.weight, self.bias)
        tracker = getattr(self, "_ps_tracker", None)
        if tracker is not None:
            tracker.observe(self._ps_finite_slot, out)
        elif not torch.isfinite(out).all():
            raise FloatingPointError("Переполнение Linear; checkpoint нельзя безусловно считать FP16-совместимым")
        return out


class RMSNorm(nn.RMSNorm):
    def reset_parameters(self):
        pass

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class Operations:
    Linear = Linear
    RMSNorm = RMSNorm


@lru_cache(maxsize=8)
def regular_hadamard(size, device):
    # Регулярная ConvRot матрица из формата Comfy Kitchen; не Sylvester H2.
    if size < 4 or size & (size-1) or (size.bit_length()-1) % 2:
        raise ValueError("ConvRot group_size должен быть степенью 4")
    h4 = torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], dtype=torch.float32, device=device)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / size ** .5


def dequantize_rows(q, scales, convrot, group_size, dtype=torch.float16, check=True):
    if q.numel() == 0:
        return q.to(dtype)  # empty trailing FSDP shard: reshape(0,-1,gs) is invalid
    w = q.float() * scales.float()
    if convrot:
        h = regular_hadamard(group_size, q.device)
        w = (w.reshape(w.shape[0], -1, group_size) @ h.T).reshape_as(w)
    if check and w.numel() and (not torch.isfinite(w).all() or w.abs().max() > torch.finfo(dtype).max):
        raise FloatingPointError("INT8 dequantization выходит за диапазон FP16")
    return w.to(dtype)


class Int8Linear(nn.Module):
    def __init__(self, src, quant, rows):
        super().__init__()
        self.in_features, self.out_features = src.in_features, src.out_features
        self.weight = nn.Parameter(torch.empty(src.weight.shape, dtype=torch.int8, device="meta"), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty((src.out_features, 1), dtype=torch.float32, device="meta"), requires_grad=False)
        self.bias = src.bias
        self._ps_original_compute_dtype = src.weight.dtype
        self.convrot, self.group_size, self.rows = quant["convrot"], quant["group_size"], rows

    def forward(self, x):
        safe = getattr(self, "_ps_safe", False)
        tracker = getattr(self, "_ps_tracker", None)
        x = x.float() if safe else x.half()
        out = torch.empty(x.shape[:-1] + (self.out_features,), dtype=x.dtype, device=x.device)
        prepared = getattr(self, "_ps_prepared", None)
        for a in range(0, self.out_features, self.rows):
            b = min(a + self.rows, self.out_features)
            if prepared is not None:
                from .fp16_safe import prepared_matmul
                value = prepared_matmul(x, prepared[a//self.rows])
                out[...,a:b] = value if self.bias is None else value+self.bias[a:b].float()
                continue
            w = dequantize_rows(self.weight[a:b], self.weight_scale[a:b], self.convrot, self.group_size,
                                dtype=torch.float32 if safe else torch.float16, check=tracker is None)
            bias = None if self.bias is None else self.bias[a:b].to(x.dtype)
            if safe and not self._ps_fp32:
                from .fp16_safe import safe_linear
                out[..., a:b] = safe_linear(x, w, bias)
            else:
                out[..., a:b] = F.linear(x, w, bias)
            del w
        if tracker is not None:
            tracker.observe(self._ps_finite_slot, out)
        elif not torch.isfinite(out).all():
            raise FloatingPointError("NaN/Inf после INT8->FP16 Linear")
        return out


def install_int8(model, quant, rows):
    import torch as _torch
    skipped = {}
    for name, conf in quant.items():
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        old = getattr(parent, leaf)
        if isinstance(old, _torch.nn.Embedding):
            # Квантованный embedding (qwen3vl int8 checkpoints): lookup-таблица,
            # не GEMM — Int8Linear здесь неприменим. Остаётся native floating
            # параметром; деквантизация локальных строк (ConvRot+row scale)
            # выполняется в load_local через quant_map.
            skipped[name] = conf
            continue
        # Duck-typing: наш Operations.Linear и нативный comfy ops Linear оба
        # подклассы nn.Linear. Int8Linear использует только in/out_features,
        # weight и bias — нативный Linear совместим как источник.
        if not isinstance(old, _torch.nn.Linear) or not hasattr(old, "in_features"):
            raise ValueError(f"INT8 ожидает Linear/Embedding (got {type(old).__name__}): {name}")
        setattr(parent, leaf, Int8Linear(old, conf, rows))
    return skipped


@contextmanager
def prepared_linears(modules, budget, enabled=True):
    """Bounded active-MLP cache; никогда не хранится между блоками/forward.

    INT8 storage/scales остаются параметрами; только данный MLP временно получает
    scaled-half tiles. Не меняет FSDP representation и не вызывает collectives.
    """
    from .fp16_safe import prepare_matmul
    bounded = [not isinstance(m, Int8Linear) and getattr(m, "_ps_weight_bounds", (None,))[0] == 1.
               for m in modules]
    size = sum(m.in_features*m.out_features*2 for m, reuse in zip(modules, bounded) if not reuse)
    staging = max((0 if reuse else 16*m.in_features*(min(m.rows,m.out_features) if isinstance(m,Int8Linear)
                    else m.out_features) for m, reuse in zip(modules, bounded)),default=0)
    use = enabled and size+staging <= budget and all(getattr(m,"_ps_safe",False) and not m._ps_fp32 for m in modules)
    info = dict(prepared_weight_bytes=size if use else 0, preparation_peak_estimate=size+staging if use else 0,
                preparation="active MLP, reused across token chunks" if use else "per-call tiles; preparation budget insufficient or one chunk")
    try:
        if use:
            for m in modules:
                if isinstance(m,Int8Linear):
                    tiles=[]
                    for a in range(0,m.out_features,m.rows):
                        b=min(a+m.rows,m.out_features)
                        w=dequantize_rows(m.weight[a:b],m.weight_scale[a:b],m.convrot,m.group_size,dtype=torch.float32,check=False)
                        tiles.append(prepare_matmul(w.T))
                        del w
                    m._ps_prepared=tiles
                else:
                    m._ps_prepared=prepare_matmul(m.weight.T, getattr(m, "_ps_weight_bounds", None))
        yield info
    finally:
        for m in modules:
            if hasattr(m,"_ps_prepared"):
                del m._ps_prepared
