"""Бэкенды блочно-разреженной attention Veda и их выбор в worker.

  triton-int8  — ядро Veda/SageAttention, INT8 Q·K (NVIDIA sm80+);
  triton-fp16  — то же обхождение, FP16 Q·K с FP32-накоплением (любой GPU, где Triton компилирует tl.dot,
                 в т.ч. V100 / sm70);
  torch-sdpa   — без Triton: оставленные тайлы ключей собираются в плотный буфер и считаются
                 torch SDPA (cuBLAS/cutlass tensor cores) с маской паддинга.

Каждый бэкенд проходит self-test против FP32-эталона на реальном GPU (и проверку, что маска действительно
соблюдается). `auto` замеряет прошедшие self-test бэкенды на задаче разумного размера и берёт самый быстрый;
при нескольких rank время суммируется all-reduce — выбор одинаков на всех GPU.
"""
import time

import torch
import torch.nn.functional as F

from . import reference
from . import selection
from . import tiling

ORDER = ("triton-int8", "triton-fp16", "torch-sdpa")


class BackendUnavailable(RuntimeError):
    pass


class Backend:
    name = "backend"
    tolerance = 0.02

    def attend(self, q, k, v, block_mask, layout, softmax_scale):
        raise NotImplementedError


class TritonInt8(Backend):
    name = "triton-int8"
    tolerance = 0.05  # INT8 Q/K со scale на блок: ~1-3% от absmax (свойство формата)

    def __init__(self, device):
        if torch.cuda.get_device_capability(device) < (8, 0):
            raise BackendUnavailable("triton-int8: нужен GPU sm80+ (INT8 tensor cores в Triton)")
        from . import kernels
        self.kernels = kernels

    def attend(self, q, k, v, block_mask, layout, softmax_scale):
        index, count = selection.tile_index_list(block_mask & layout.kv_ok)
        return self.kernels.attend_int8(q, k, v, index, count, layout.valid_count, softmax_scale)


class TritonFP16(Backend):
    name = "triton-fp16"
    tolerance = 0.02

    def __init__(self, device):
        from . import kernels
        self.kernels = kernels

    def attend(self, q, k, v, block_mask, layout, softmax_scale):
        index, count = selection.tile_index_list(block_mask & layout.kv_ok)
        return self.kernels.attend_fp16(q, k, v, index, count, layout.valid_count, softmax_scale)


def _real_slots(layout):
    cached = getattr(layout, "_ps_real_slots", None)
    if cached is None:
        cached = torch.nonzero(layout.slot_valid.bool()).view(-1)
        layout._ps_real_slots = cached
    return cached


class TorchSDPA(Backend):
    """Сборка оставленных тайлов + SDPA. Глобальные строки (текст/аудио) — плотная SDPA по всем реальным строкам."""
    name = "torch-sdpa"
    tolerance = 0.02
    chunk_bytes = 768 * 2**20

    def __init__(self, device):
        self.device = device

    def attend(self, q, k, v, block_mask, layout, softmax_scale):
        tile = tiling.TILE_SIZE
        heads, dim = q.shape[1], q.shape[2]
        if q.device.type != "cuda":
            dtype = torch.float32  # CPU (тесты): fp16 SDPA на CPU есть не во всех сборках
        else:
            dtype = torch.float16 if q.dtype not in (torch.float16, torch.bfloat16) else q.dtype
        q, k, v = (t.to(dtype) for t in (q, k, v))
        out = torch.zeros_like(q)
        n_video = layout.n_video_tiles
        kh, vh = k.transpose(0, 1), v.transpose(0, 1)                       # [H', N, D] (view)
        if layout.n_global_tiles:
            real = _real_slots(layout)
            qg = q[n_video * tile:].transpose(0, 1).unsqueeze(0)            # [1, H', Lg, D]
            kr = kh.index_select(1, real).unsqueeze(0)
            vr = vh.index_select(1, real).unsqueeze(0)
            og = F.scaled_dot_product_attention(qg, kr, vr, scale=softmax_scale)
            out[n_video * tile:] = og[0].transpose(0, 1)
            del kr, vr, og
        if n_video == 0:
            return out
        index, count = selection.tile_index_list(block_mask[:, :n_video] & layout.kv_ok)
        widest = index.shape[-1]
        per_tile = 2 * heads * widest * tile * dim * q.element_size() + heads * widest * tile * tile * 4
        if q.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(q.device)
            budget = max(64 * 2**20, min(self.chunk_bytes, free // 4))
        else:
            budget = self.chunk_bytes
        step = max(1, int(budget // max(1, per_tile)))
        offsets = torch.arange(tile, device=q.device)
        heads_index = torch.arange(heads, device=q.device)[:, None, None]
        for c0 in range(0, n_video, step):
            c1 = min(n_video, c0 + step)
            cnt = count[:, c0:c1]
            width = int(cnt.max().item())
            if width == 0:
                continue
            idx = index[:, c0:c1, :width].long()                             # [H', C, W]
            slots = (idx[..., None] * tile + offsets).flatten(-2)             # [H', C, W*128]
            valid = ((torch.arange(width, device=q.device)[None, None, :] < cnt[..., None])[..., None]
                     & (offsets < layout.valid_count[idx][..., None])).flatten(-2)
            kg = kh[heads_index, slots]                                      # [H', C, L, D]
            vg = vh[heads_index, slots]
            chunk = c1 - c0
            qc = q[c0 * tile:c1 * tile].view(chunk, tile, heads, dim).permute(2, 0, 1, 3)
            o = F.scaled_dot_product_attention(
                qc.reshape(heads * chunk, 1, tile, dim), kg.reshape(heads * chunk, 1, -1, dim),
                vg.reshape(heads * chunk, 1, -1, dim), attn_mask=valid.reshape(heads * chunk, 1, 1, -1),
                scale=softmax_scale)
            out[c0 * tile:c1 * tile] = o.view(heads, chunk, tile, dim).permute(1, 2, 0, 3).reshape(chunk * tile,
                                                                                                 heads, dim)
            del kg, vg, o, valid, slots
        return out.nan_to_num_(0.0)


CLASSES = {"triton-int8": TritonInt8, "triton-fp16": TritonFP16, "torch-sdpa": TorchSDPA}


# ------------------------------------------------------------- self-test / bench
def problem(device, dtype, target=(5, 8, 8), shape=(4, 4, 8), heads=2, budget=2, seed=0):
    """Малая задача со всеми трудными случаями: референс- и целевой спаны, неполные тайлы, глобальные строки."""
    spans = [tiling.TiledSpan(70, (1, 8, 16), tiling.TileShape(1, 8, 16)),
             tiling.TiledSpan(238, target, tiling.TileShape(*shape))]
    seq_len = 238 + target[0] * target[1] * target[2] + 110
    layout = tiling.build_tile_layout(spans, seq_len, device)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(3, seq_len, heads, 128, generator=generator) * 1.5
    q, k, v = (tiling.gather_tiles(t.to(device=device, dtype=dtype), layout, torch.arange(heads, device=device))
               for t in x)
    scores = torch.randn(heads, layout.n_video_tiles, layout.n_video_tiles, generator=generator).to(device)
    blocks = selection.column_blocks(layout, selection.Budget(tiles=budget), selection.Budget(tiles=1))
    index, keep = selection.select(scores, layout, blocks)
    return q, k, v, selection.block_mask(index, keep, layout), layout


def self_test(backend, device):
    q, k, v, mask, layout = problem(device, torch.float16)
    scale = 128 ** -0.5 * 2.0  # не стандартный scale: проверяем явную передачу (FP16 Safe)
    try:
        out = backend.attend(q, k, v, mask, layout, scale)
    except BackendUnavailable:
        raise
    except Exception as error:
        raise BackendUnavailable(f"{backend.name}: self-test упал: {type(error).__name__}: {error}") from error
    real = layout.slot_valid.bool()
    want = reference.block_sparse_attention(q, k, v, mask, layout, scale)[real]
    dense = reference.block_sparse_attention(q, k, v, torch.ones_like(mask) & layout.kv_ok, layout, scale)[real]
    got = out[real].float()
    if not torch.isfinite(got).all():
        raise BackendUnavailable(f"{backend.name}: не конечные значения в self-test")
    err = (got - want).abs().max().item()
    gap = (dense - want).abs().max().item()
    if (got - dense).abs().max().item() < 0.5 * gap:
        raise BackendUnavailable(f"{backend.name}: игнорирует блочную маску (считает плотно)")
    limit = backend.tolerance * max(1.0, want.abs().max().item())
    if not err <= limit:
        raise BackendUnavailable(f"{backend.name}: неверный результат (ошибка {err:.3g}, допуск {limit:.3g})")
    return err


def benchmark(backend, device, repeats=3):
    """Время на задаче ~16k токенов (4 головы, 32 тайла на строку) — порядок как у реального шага."""
    q, k, v, mask, layout = problem(device, torch.float16, target=(16, 24, 42), shape=(2, 8, 8), heads=4,
                                    budget=32, seed=1)
    scale = 128 ** -0.5
    backend.attend(q, k, v, mask, layout, scale)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        backend.attend(q, k, v, mask, layout, scale)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / repeats


def candidates(device, choice):
    """auto на sm<80: только torch-sdpa — Triton 3.x на Volta считает tl.dot без tensor cores (FMA) и проиграет
    cutlass-SDPA; triton-fp16 можно выбрать явно для проверки."""
    if choice != "auto":
        if choice not in CLASSES:
            raise ValueError(f"Veda backend: {' / '.join(('auto',) + ORDER)}")
        return (choice,)
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        return ("torch-sdpa",)
    return ORDER


def warmup(backend, device):
    """Явно выбранный бэкенд: только компиляция/первый запуск на малой задаче, без сравнения с эталоном."""
    q, k, v, mask, layout = problem(device, torch.float16)
    out = backend.attend(q, k, v, mask, layout, 128 ** -0.5)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out


def evaluate(device, choice="auto"):
    """Локально (без коллективов). -> (созданные, отчёт, времена по ORDER).

    Явный выбор: бэкенд создаётся и компилируется, self-test не проводится (пользователь выбрал сам).
    auto: self-test + замер; не прошедший кандидат просто не участвует в выборе (причина — в отчёте).
    """
    names = candidates(device, choice)
    explicit = choice != "auto"
    measure = len(names) > 1
    attempts, created, timings = {}, {}, []
    for name in ORDER:
        if name not in names:
            timings.append(float("inf"))
            continue
        try:
            backend = CLASSES[name](device)
            if explicit:
                warmup(backend, device)
                attempts[name] = "ok (выбран явно)"
                seconds = 0.0
            else:
                error = self_test(backend, device)
                seconds = benchmark(backend, device) if measure else 0.0
                attempts[name] = f"ok (ошибка {error:.2g}" + (f", {seconds * 1e3:.1f} мс)" if measure else ")")
            created[name] = backend
            timings.append(seconds)
        except Exception as error:  # ImportError Triton, ошибка компиляции, self-test
            attempts[name] = f"{type(error).__name__}: {error}"[:400]
            timings.append(float("inf"))
    return created, attempts, timings
