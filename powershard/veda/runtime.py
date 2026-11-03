"""Veda в worker PowerShard (MiniMax H3, kind h3q).

* Каждый DiT-блок H3: `attn._ps_attention` (диспетчер attention PowerShard) оборачивается VedaDispatch, а
  `attn.forward` запоминает transformer_options вызова (block_index, minimax_h3_layout, sigmas).
* Настройки приходят в каждом RPC forward (`_powershard_veda` от host-ноды): смена разреженности не
  перезапускает workers. Предиктор загружается один раз на процесс (кэш по пути и mtime).
* Ulysses: после all-to-all у rank полная последовательность для голов [r*h, r*h+h) — Veda считается
  локально. В token-режиме (локальные строки q) вызов идёт плотным путём; host-нода требует ulysses.
* Выбор бэкенда выполняется в call() на всех rank: явно выбранный только компилируется (по очереди rank-ов),
  auto — self-test + замер; отчёты собираются со всех GPU, бэкенд одинаков на всех.
"""
import collections
import os
import sys

import torch
import torch.distributed as dist

from . import settings as veda_settings

_BUNDLES = {}
_BACKENDS = {}  # (device, путь предиктора, выбор) -> (бэкенд, отчёт); согласовано между rank


def load_bundle_cached(path):
    from .bundle import load_bundle
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key not in _BUNDLES:
        _BUNDLES.clear()  # один предиктор на процесс (~0.3 GB RAM)
        _BUNDLES[key] = load_bundle(path)
    return _BUNDLES[key]


def _report_lines(reports):
    return "\n".join(f"  rank {rank}: " + ("; ".join(f"{k}: {v}" for k, v in attempts.items()) or "-")
                     + (f"; ошибка: {error}" if error else "") for rank, (error, _, attempts) in enumerate(reports))


def prepare_collectively(device, path, choice, geometry):
    """Предиктор + бэкенд на всех rank.

    * Ranks готовятся ПО ОЧЕРЕДИ (rank 0, затем 1, ...): Triton компилирует ядро один раз, остальные берут его
      из кэша — шесть процессов не пишут в ~/.triton/cache одновременно.
    * Ошибки ловятся локально, барьеры и обмен отчётами выполняются всегда — ни один rank не зависнет в NCCL.
    * Отчёты всех rank собираются: в ошибке видно, на каком GPU и почему не получилось.
    """
    from . import backends
    key = (str(device), path, choice)
    if key in _BACKENDS:
        return load_bundle_cached(path), _BACKENDS[key]
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    error, bundle = None, None
    created, attempts, timings = {}, {}, [float("inf")] * len(backends.ORDER)
    for turn in range(world):
        if turn == rank:
            try:
                bundle = load_bundle_cached(path)
                found = (bundle.num_layers, bundle.num_heads, bundle.head_dim)
                if found != geometry:
                    raise ValueError(f"предиктор Veda обучен для {found} (блоки x головы x head_dim), "
                                     f"модель H3 — {geometry}")
                created, attempts, timings = backends.evaluate(device, choice)
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"[:600]
        if distributed:
            dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
    local = (error, timings, attempts)
    if distributed:
        reports = [None] * world
        dist.all_gather_object(reports, local)
    else:
        reports = [local]
    if any(report[0] for report in reports):
        raise RuntimeError("Veda не запустилась:\n" + _report_lines(reports))
    total = [sum(report[1][i] for report in reports) for i in range(len(backends.ORDER))]
    best = min(range(len(total)), key=lambda i: total[i])
    if total[best] == float("inf"):
        what = "выбранный бэкенд не запустился" if choice != "auto" else "ни один бэкенд не прошёл проверку"
        raise RuntimeError(f"Veda: {what} на всех GPU:\n" + _report_lines(reports))
    name = backends.ORDER[best]
    if choice == "auto" and rank == 0 and any(v for r in reports for k, v in r[2].items() if not v.startswith("ok")):
        print("Veda auto: выбран " + name + "; не прошли проверку:\n" + _report_lines(reports), file=sys.stderr)
    if (name == "triton-fp16" and rank == 0 and device.type == "cuda"
            and torch.cuda.get_device_capability(device) < (8, 0)):
        print("WARNING: Veda triton-fp16 на sm70 (V100): Triton 3.x не использует tensor cores Volta (MMAv1 удалён), "
              "tl.dot считается на FP32-ядрах — медленно и при неполной мощности GPU. Используйте backend=torch-sdpa "
              "(или auto)", file=sys.stderr)
    chosen = (created[name], attempts)
    _BACKENDS[key] = chosen
    return bundle, chosen


class VedaDispatch:
    """Обёртка диспетчера attention одного DiT-блока: разреженный путь или исходный."""

    def __init__(self, inner, runtime, attn):
        self.inner, self.runtime, self.attn = inner, runtime, attn

    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)

    def __call__(self, q, k, v, options=None, group="unspecified", compute_fp16=False):
        runtime = self.runtime
        reason = runtime.decline_reason(q, k, v, options)
        if reason is not None:
            if reason != "off":
                runtime.counts[reason] += 1
            return self.inner(q, k, v, options, group=group, compute_fp16=compute_fp16)
        return runtime.sparse(q, k, v, options, self.attn)


class VedaRuntime:
    def __init__(self, net, device):
        self.device = device
        self.blocks = len(net.blocks)
        attn0 = net.blocks[0].attn
        self.heads, self.head_dim = attn0.heads, attn0.head_dim
        self.active = False
        self.context = {}
        self.engine = None
        self.config = None
        self.backend_attempts = {}
        self.counts = collections.Counter()
        self.announced = set()
        self.spec = None
        self._steps = {}
        for block in net.blocks:
            attn = block.attn
            attn._ps_attention = VedaDispatch(attn._ps_attention, self, attn)
            original = attn.forward

            def forward(x, rope_freqs=None, transformer_options=None, _original=original):
                self.context = transformer_options if isinstance(transformer_options, dict) else {}
                return _original(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
            attn.forward = forward

    # ------------------------------------------------------------------ per RPC
    def begin(self, spec):
        """spec от host (или None — Veda выключена для этого вызова)."""
        self.counts = collections.Counter()
        self._steps.clear()
        if not spec:
            self.active = False
            return
        from . import selection
        from .engine import VedaEngine
        bundle, (backend, attempts) = prepare_collectively(self.device, spec["path"], spec.get("backend", "auto"),
                                                           (self.blocks, self.heads, self.head_dim))
        self.backend_attempts = attempts
        key = (spec["path"], spec.get("size"), spec.get("mtime_ns"), id(bundle), repr(spec["generated"]),
               repr(spec["reference"]), backend.name)
        if self.engine is None or self.config is None or self.config[0] != key:
            self.engine = VedaEngine(bundle, selection.Budget(**spec["generated"]),
                                     selection.Budget(**spec["reference"]), backend, self.device)
        from .engine import Stats
        self.engine.stats = Stats()
        self.config = (key, frozenset(spec.get("dense_layers", ())), frozenset(spec.get("dense_steps", ())))
        self.active = True

    def report(self):
        if not self.active or self.engine is None:
            return None
        out = dict(backend=self.engine.backend.name, calls=dict(self.counts), stats=self.engine.stats.report(),
                   chunking=dict(self.engine.chunking))
        if any(not str(v).startswith("ok") for v in self.backend_attempts.values()):
            out["backend_attempts"] = self.backend_attempts
        return out

    # ----------------------------------------------------------------- decision
    def _step(self, options):
        sigmas, schedule = options.get("sigmas"), options.get("sample_sigmas")
        if sigmas is None or schedule is None:
            return None
        key = id(sigmas)  # тензоры одного RPC; кэш очищается в begin() — одна синхронизация на forward
        if key not in self._steps:
            self._steps[key] = veda_settings.step_index(float(sigmas.flatten()[0]), schedule.flatten().tolist())
        return self._steps[key]

    def decline_reason(self, q, k, v, options):
        if not self.active:
            return "off"
        ctx = self.context
        layer = ctx.get("block_index")
        _, dense_layers, dense_steps = self.config
        if not isinstance(layer, int) or layer >= self.blocks:
            return "слой вне предиктора"
        if layer in dense_layers:
            return "full-attention слой"
        if dense_steps and self._step(ctx) in dense_steps:
            return "full-attention шаг"
        if self.engine.generated.keeps_all and self.engine.reference.keeps_all:
            return "разреженность 0%"
        layout = ctx.get("minimax_h3_layout")
        if layout is None:
            return "нет H3 layout"
        if q.dim() != 4 or q.shape[0] != 1 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
            return "token-режим или другая форма q/k/v (нужен sequence_mode=ulysses)"
        if getattr(layout, "seq_len", None) != q.shape[1]:
            return "layout другого прохода"
        from .h3_layout import LayoutError
        try:
            self.spec = self.engine.layout_spec(layout)
        except LayoutError as error:
            return f"layout не читается ({error})"[:120]
        if options is not None and (getattr(options, "causal", False) or getattr(options, "window_size", (-1, -1))
                                    != (-1, -1)):
            return "маска/окно attention"
        return None

    # ------------------------------------------------------------------- sparse
    def sparse(self, q, k, v, options, attn):
        engine = self.engine
        dim = q.shape[-1]
        scale = getattr(options, "softmax_scale", None) or dim ** -0.5
        q0, k0, v0 = q[0], k[0], v[0]                                       # [S, h, D]
        restore = None
        if any(t.dtype == torch.float32 for t in (q0, k0, v0)):
            # FP16 Safe + math-диспетчер: q/k/v в FP32. Нормируем степенями двойки, как Safe-путь attention.
            from ..fp16_safe import power2_scale
            # q/k — к ~128 (как Safe qk scales: math-SDPA умножает их на sqrt(scale) в fp16), V — к 16384.
            sq, sk = (power2_scale(t.float().abs().amax() / 128.) for t in (q0, k0))
            sv = power2_scale(v0.float().abs().amax() / 16384.)
            q0, k0, v0 = (q0.float() / sq).half(), (k0.float() / sk).half(), (v0.float() / sv).half()
            scale = scale * float(sq * sk)
            restore = sv
        state = getattr(attn, "_ps_sequence", None) or {}
        local = q0.shape[1]
        if state.get("enabled") and state.get("ulysses") and dist.is_initialized():
            heads_global = dist.get_rank() * local + torch.arange(local, device=q0.device)
        else:
            heads_global = torch.arange(local, device=q0.device)
        spec = self.spec
        choice = engine.plan_for(spec)
        self._announce(spec, choice)
        out = engine.attention(q0, k0, v0, self.context["block_index"], spec, choice.plan, heads_global, float(scale))
        self.counts["sparse"] += 1
        if restore is not None:
            out = out.float() * restore
        return out.unsqueeze(0)

    def _announce(self, spec, choice):
        key = (spec.target.grid, choice.plan.name)
        if key in self.announced:
            return
        self.announced.add(key)
        if not dist.is_initialized() or dist.get_rank() == 0:
            from .plans import describe_grid
            print(f"Veda: backend {self.engine.backend.name}; видео {describe_grid(spec.target.grid)}; план "
                  f"{choice.how}; референсов {len(spec.references)}"
                  + ("" if choice.exact else " (размер вне обучения предиктора: проверьте качество)"),
                  file=sys.stderr)
