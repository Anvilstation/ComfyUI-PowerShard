"""Обычные tensors, FP32 norm и переносимый INT8 ConvRot без tensor subclasses."""
from functools import lru_cache
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Linear):
    def reset_parameters(self):
        pass  # Только checkpoint loader; создание исключительно на meta.

    def forward(self, x):
        prepared = getattr(self, "_ps_prepared", None)
        rows = getattr(self, "_ps_output_rows", 0)
        if rows and self.out_features > rows:
            # Bound weight conversion/scaling temporaries. FSDP still gathers
            # this module once; tiles are ordinary GEMMs, never collectives.
            safe = getattr(self, "_ps_safe", False)
            out = torch.empty(x.shape[:-1]+(self.out_features,), device=x.device,
                              dtype=torch.float32 if safe else self.weight.dtype)
            for a in range(0, self.out_features, rows):
                b = min(a+rows, self.out_features)
                weight = self.weight[a:b]
                bias = None if self.bias is None else self.bias[a:b]
                if safe and not self._ps_fp32:
                    from .fp16_safe import safe_linear
                    out[...,a:b] = safe_linear(x, weight, bias)
                elif safe:
                    out[...,a:b] = F.linear(x.float(), weight.float(), None if bias is None else bias.float())
                else:
                    out[...,a:b] = F.linear(x.to(weight.dtype), weight, bias)
        elif getattr(self, "_ps_safe", False):
            if self._ps_fp32:
                out = F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())
            else:
                from .fp16_safe import safe_linear
                if prepared is None:
                    out = safe_linear(x, self.weight, self.bias)
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
    if q.shape[0] == 0:
        return q.to(dtype)
    w = q.float() * scales.float()
    if convrot:
        h = regular_hadamard(group_size, q.device)
        w = (w.reshape(w.shape[0], -1, group_size) @ h.T).reshape_as(w)
    if check and (not torch.isfinite(w).all() or w.abs().max() > torch.finfo(dtype).max):
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
    size = sum(m.in_features*m.out_features*2 for m in modules)
    staging = max((16*m.in_features*(min(m.rows,m.out_features) if isinstance(m,Int8Linear)
                    else m.out_features) for m in modules),default=0)
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
                    m._ps_prepared=prepare_matmul(m.weight.T)
        yield info
    finally:
        for m in modules:
            if hasattr(m,"_ps_prepared"):
                del m._ps_prepared
