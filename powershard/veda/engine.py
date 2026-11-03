# Переработано из Veda-on-ComfyUI (MIT, (c) 2026 veda-sparse), veda_comfy/core/engine.py. См. NOTICE.md.
"""Один вызов Veda attention для группы голов этого rank.

Отличия от официального движка: голова задаётся ГЛОБАЛЬНЫМ индексом (Ulysses даёт rank-у срез голов
[r*h, r*h+h), последние — нулевые паддинг-головы), softmax scale передаётся явно (FP16 Safe делит q/k на
степени двойки), проекции предиктора своих голов держатся на GPU (без H2D на каждый слой).
Выбор тайлов инвариантен к положительному масштабу q и k: логиты блоков масштабируются константой на голову.
"""
import weakref

import torch

from . import h3_layout
from . import predictor
from . import selection
from . import tiling

_CHUNK_SHARE = 8
_MIN_CHUNK_BYTES = 128 * 2**20
_MAX_CHUNK_BYTES = 1024 * 2**20
_WORKING_SET_SHARE = 6


class Stats:
    """Накопление на устройстве, чтение один раз за задачу."""

    def __init__(self):
        self.kept = None
        self.video_pairs = 0
        self.work = None
        self.dense_work = 0.0
        self.calls = 0

    def add(self, keep, layout):
        heads = keep.shape[0]
        kept = keep.sum()
        work = kept + heads * layout.n_global_tiles * (layout.n_tiles + layout.n_video_tiles)
        self.kept = kept if self.kept is None else self.kept + kept
        self.work = work if self.work is None else self.work + work
        self.video_pairs += heads * layout.n_video_tiles ** 2
        self.dense_work += heads * (layout.seq_len / tiling.TILE_SIZE) ** 2

    def report(self):
        if self.kept is None:
            return dict(calls=self.calls)
        return dict(calls=self.calls, video_tiles_kept=round(float(self.kept) / max(1, self.video_pairs), 4),
                    attention_computed=round(min(1.0, float(self.work) / max(1e-9, self.dense_work)), 4))


class VedaEngine:
    def __init__(self, bundle, generated, reference, backend, device):
        self.bundle, self.generated, self.reference = bundle, generated, reference
        self.backend, self.device = backend, device
        self.stats = Stats()
        self.chunking = {}
        self._specs = weakref.WeakKeyDictionary()
        self._tile_layouts = {}
        self._weights = {}

    # -- кэши --
    def layout_spec(self, layout):
        try:
            spec = self._specs.get(layout)
        except TypeError:
            return h3_layout.describe(layout)
        if spec is None:
            spec = h3_layout.describe(layout)
            self._specs[layout] = spec
        return spec

    def plan_for(self, spec):
        return self.bundle.plans.select(spec.target.grid)

    def _tile_layout(self, spec, shape):
        tile_references = not self.reference.keeps_all
        key = (spec, shape, tile_references)
        layout = self._tile_layouts.get(key)
        if layout is None:
            if len(self._tile_layouts) >= 64:  # разные разрешения/длины за долгую сессию
                self._tile_layouts.clear()
            spans = []
            if tile_references:
                spans = [tiling.TiledSpan(s.start, s.grid, tiling.least_padding_shape(s.grid))
                         for s in spec.references]
            spans.append(tiling.TiledSpan(spec.target.start, spec.target.grid, shape))
            layout = tiling.build_tile_layout(spans, spec.seq_len, self.device)
            self._tile_layouts[key] = layout
        return layout

    def weights(self, layer, heads_key):
        """Проекции голов (кортеж глобальных индексов) слоя на GPU; набор голов rank не меняется."""
        key = (layer, heads_key)
        cached = self._weights.get(key)
        if cached is None:
            index = torch.tensor(heads_key, dtype=torch.long)
            cached = (self.bundle.proj_q[layer].index_select(0, index).to(self.device),
                      self.bundle.proj_k[layer].index_select(0, index).to(self.device))
            self._weights[key] = cached
        return cached

    def reset(self):
        self._tile_layouts.clear()
        self._specs = weakref.WeakKeyDictionary()
        self.stats = Stats()

    def _chunk_bytes(self):
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            self.chunking["free_bytes"] = free
            return int(min(max(free // _CHUNK_SHARE, _MIN_CHUNK_BYTES), _MAX_CHUNK_BYTES))
        return 256 * 2**20

    # -- вызов --
    @torch.no_grad()
    def attention(self, q, k, v, layer, spec, plan, heads_global, softmax_scale):
        """q, k, v: [S, h, D] (полная последовательность, h голов этого rank, после norm/RoPE/Safe-масштаба).

        heads_global: [h] long — глобальный индекс каждой локальной головы (>= num_heads: паддинг Ulysses).
        Возвращает [S, h, D] в dtype q (для паддинг-голов — нули).
        """
        seq_len, heads, dim = q.shape
        dtype = q.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            q, k, v = (t.to(torch.float16) for t in (q, k, v))
        out = q.new_zeros(seq_len + 1, heads, dim)
        chunk_bytes = self._chunk_bytes()
        local = torch.arange(heads, device=self.device)
        heads_cpu = heads_global.tolist()            # одна синхронизация на вызов
        heads_global = heads_global.to(self.device)
        # Рабочее множество — от числа голов МОДЕЛИ: при Ulysses у rank лишь часть голов, и граница от
        # собственного выхода дробила бы вызов по одной голове (много мелких запусков и синхронизаций).
        total_heads = max(heads, self.bundle.num_heads)
        chunks = 0
        for group in plan.head_groups(layer, self.device):
            member = (heads_global[:, None] == group.heads[None, :]).any(1)
            local_heads = local[member]
            if local_heads.numel() == 0:
                continue
            layout = self._tile_layout(spec, group.shape)
            blocks = selection.column_blocks(layout, self.generated, self.reference)
            per_head = layout.num_slots * dim * q.element_size()
            step = max(1, chunk_bytes // per_head)
            working = seq_len * total_heads * dim * out.element_size()
            step = min(step, max(1, working // (per_head * _WORKING_SET_SHARE)))
            local_list = local_heads.tolist()
            for start in range(0, len(local_list), step):
                ids = local_list[start:start + step]
                heads_chunk = torch.tensor(ids, dtype=torch.long, device=self.device)
                chunks += 1
                proj_q, proj_k = self.weights(layer, tuple(heads_cpu[i] for i in ids))
                q_t = tiling.gather_tiles(q, layout, heads_chunk)
                k_t = tiling.gather_tiles(k, layout, heads_chunk)
                scores = predictor.tile_logits(predictor.pool_video_tiles(q_t, layout),
                                               predictor.pool_video_tiles(k_t, layout), proj_q, proj_k)
                index, keep = selection.select(scores, layout, blocks)
                del scores
                self.stats.add(keep, layout)
                mask = selection.block_mask(index, keep, layout)
                v_t = tiling.gather_tiles(v, layout, heads_chunk)
                o_t = self.backend.attend(q_t, k_t, v_t, mask, layout, softmax_scale)
                del q_t, k_t, v_t, mask
                tiling.scatter_tiles_(out, o_t.to(out.dtype), layout, heads_chunk)
                del o_t
        self.stats.calls += 1
        self.chunking["chunks_per_layer"] = chunks
        out = out[:seq_len]
        return out if out.dtype == dtype else out.to(dtype)
