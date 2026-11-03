"""Ускорители и guidance для распределённых Wan/LTX (host + worker).

Host:
  * допуск host-only wrappers сообщества, которые не трогают веса и forward модели:
    EasyCache / LazyCache (comfy_extras.nodes_easycache), Context Windows (comfy.context_windows),
    sampler_*_cfg_function (STG, Modality Guidance, NAG-подобные пост-CFG узлы);
  * исполнение DIFFUSION_MODEL wrappers (EasyCache) вокруг RPC — как native forward;
  * holders для PowerShard Block Cache / NAG / RIFLEx (лежат в transformer_options, в workers
    уходят только числа и staged tensors).
Worker:
  * FirstBlockCache — «TeaCache/FBCache»: блок 0 считается всегда, остальные пропускаются, если
    относительное изменение его residual меньше порога; решение принимается по all-reduce
    сумм, поэтому одинаково на всех rank (FSDP коллективы не расходятся);
  * nag_combine — Normalized Attention Guidance над выходом cross-attention;
  * riflex_embedder — RIFLEx: одна временная частота RoPE Wan ставится в 0.9·2π/L.
"""
import json
import math

# --------------------------------------------------------------------------- host
HOST_SAFE_MODULES = ("comfy_extras.nodes_easycache", "comfy.context_windows")
HOST_SAFE_MODEL_OPTIONS = ("sampler_post_cfg_function", "sampler_pre_cfg_function", "sampler_cfg_function",
                           "disable_cfg1_optimization", "context_handler",
                           # Skip Layer Guidance (Simple): host вызывает модель дважды; пропуск блоков — см. image_adapter
                           "sampler_calc_cond_batch_function")
HOST_ONLY_TRANSFORMER_KEYS = ("easycache", "context_window", "powershard_block_cache", "powershard_nag",
                              "powershard_riflex")


def _function_module(fn):
    target = getattr(fn, "func", fn)  # functools.partial
    return getattr(target, "__module__", "") or ""


def is_host_safe(fn):
    if getattr(getattr(fn, "func", fn), "_powershard_host_safe", False):
        return True
    module = _function_module(fn)
    return any(module == m or module.endswith("." + m) for m in HOST_SAFE_MODULES)


def strip_host_safe_wrappers(wrappers):
    """Удаляет wrappers известных host-only ускорителей; остальное — в обычную проверку."""
    out = {}
    for kind, keyed in (wrappers or {}).items():
        for key, functions in (keyed or {}).items():
            kept = [f for f in functions if not is_host_safe(f)]
            if kept:
                out.setdefault(kind, {})[key] = kept
    return out


def host_safe_model_options(model_options):
    """model_options без host-only ключей (post-cfg, context handler EasyCache и т.п.)."""
    return {k: v for k, v in model_options.items() if k not in HOST_SAFE_MODEL_OPTIONS}


def check_easycache_holder(value):
    name = type(value).__name__
    module = type(value).__module__ or ""
    if not (module.endswith("nodes_easycache") and name in ("EasyCacheHolder", "LazyCacheHolder")):
        raise ValueError(f"transformer_options['easycache'] неизвестного типа {module}.{name}")


def run_diffusion_wrappers(function, owner, args, kwargs, transformer_options):
    """WrappersMP.DIFFUSION_MODEL (EasyCache и др.) вокруг удалённого forward, как в native forward."""
    import comfy.patcher_extension as extension
    wrappers = extension.get_all_wrappers(extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
    if not wrappers:
        return function(*args, **kwargs)
    return extension.WrapperExecutor.new_class_executor(function, owner, wrappers).execute(*args, **kwargs)


def current_sigma(transformer_options, timestep=None):
    sigmas = transformer_options.get("sigmas")
    if sigmas is not None:
        return float(sigmas.detach().float().max())
    return None if timestep is None else float(timestep.detach().float().max()) / 1000.


def call_key(transformer_options, shapes, extra=()):
    """Ключ записи кэша: состав batch (cond/uncond), формы и флаги прохода (STG/модальности/окно)."""
    window = transformer_options.get("context_window")
    index = list(getattr(window, "index_list", []) or [])
    value = dict(cond=list(transformer_options.get("cond_or_uncond", [])), shapes=[list(s) for s in shapes],
                 window=[index[0], index[-1], len(index)] if index else None, extra=list(extra))
    return json.dumps(value, sort_keys=True, default=str)


class BlockCacheHolder:
    """PowerShard Block Cache (FBCache/TeaCache-подобный пропуск блоков в workers)."""

    def __init__(self, threshold, sigma_start, sigma_end, max_skips, warmup_steps, max_cache_mb=1024):
        self.threshold, self.sigma_start, self.sigma_end = float(threshold), float(sigma_start), float(sigma_end)
        self.max_skips, self.warmup_steps = int(max_skips), int(warmup_steps)
        self.max_cache_mb = int(max_cache_mb)
        self.last_sigma = None
        self.steps = 0
        self.run_index = 0

    def clone(self):
        return BlockCacheHolder(self.threshold, self.sigma_start, self.sigma_end, self.max_skips, self.warmup_steps,
                                self.max_cache_mb)

    def begin_run(self):
        """Начало sampling (outer_sample): новая задача/стадия не наследует счётчики прошлой."""
        self.last_sigma = None
        self.steps = 0

    def spec(self, transformer_options, timestep, shapes, extra=(), scope=""):
        sigma = current_sigma(transformer_options, timestep)
        if sigma is not None:
            if self.last_sigma is None or sigma > self.last_sigma + 1e-6:
                self.steps = 0                     # новый sampling pass (или 2-я стадия pipeline)
                self.run_index += 1
            elif sigma < self.last_sigma - 1e-6:
                self.steps += 1
            self.last_sigma = sigma
        active = sigma is None or (self.sigma_end <= sigma <= self.sigma_start)
        # scope (эксперт MoE и т.п.) входит в префикс: записи другого прохода/эксперта workers удаляют сразу.
        prefix = f"{self.run_index}|{scope}|"
        return dict(active=bool(active and self.steps >= self.warmup_steps), threshold=self.threshold,
                    max_skips=self.max_skips, prefix=prefix, max_bytes=self.max_cache_mb << 20,
                    key=prefix + call_key(transformer_options, shapes, extra))


class NAGHolder:
    """Normalized Attention Guidance: отрицательный prompt в cross-attention положительной ветки."""

    def __init__(self, context, scale, tau, alpha, sigma_start, sigma_end, audio=True, unprocessed=False):
        self.context, self.unprocessed = context, bool(unprocessed)
        self.scale, self.tau, self.alpha = float(scale), float(tau), float(alpha)
        self.sigma_start, self.sigma_end, self.audio = float(sigma_start), float(sigma_end), bool(audio)
        self.processed = {}

    def active(self, transformer_options, timestep):
        sigma = current_sigma(transformer_options, timestep)
        return sigma is None or self.sigma_end <= sigma <= self.sigma_start

    def params(self):
        return dict(scale=self.scale, tau=self.tau, alpha=self.alpha, audio=self.audio)


class RIFLExHolder:
    def __init__(self, k, train_frames):
        self.k, self.train_frames = int(k), int(train_frames)

    def params(self):
        return dict(k=self.k, train_frames=self.train_frames)


def install_holder(patcher, key, holder):
    result = patcher.clone()
    result.model_options.setdefault("transformer_options", {})[key] = holder
    return result


# ------------------------------------------------------------------------- worker
class FirstBlockCache:
    """Кэш residual блоков 1..N для текущей задачи (очищается end_run/idle).

    entries[key] = dict(r1=[residual блока 0 по потокам], residual=[x_N - x_1 по потокам], skips, bytes).
    Записи прошлых проходов/экспертов удаляются при первом же вызове нового прохода (даже неактивном),
    общий объём ограничен spec["max_bytes"] (LRU) — Context Windows / CFG-части не копят VRAM.
    """

    def __init__(self):
        self.entries = {}
        self.stats = dict(calls=0, skipped=0, computed=0, evicted=0)

    def report(self):
        return dict(self.stats, entries=len(self.entries), bytes=self.bytes())

    def bytes(self):
        return sum(e.get("bytes", 0) for e in self.entries.values())

    def clear(self):
        self.entries.clear()
        self.stats = dict(calls=0, skipped=0, computed=0, evicted=0)

    def prune(self, spec):
        prefix = spec.get("prefix")
        if prefix and any(not k.startswith(prefix) for k in self.entries):
            self.entries = {k: v for k, v in self.entries.items() if k.startswith(prefix)}

    def decide(self, spec, r1, weights):
        """r1 — список residual блока 0 (локальные строки); weights — множители (1/world для реплицированных)."""
        import torch
        import torch.distributed as dist
        self.stats["calls"] += 1
        self.prune(spec)
        entry = self.entries.get(spec["key"])
        if entry is None or entry["skips"] >= spec["max_skips"] or len(entry["r1"]) != len(r1) \
                or any(a.shape != b.shape for a, b in zip(entry["r1"], r1)):
            return False, entry
        self.entries[spec["key"]] = self.entries.pop(spec["key"])  # LRU: недавно использованная
        device = r1[0].device
        sums = torch.zeros(2, dtype=torch.float64, device=device)
        for now, before, w in zip(r1, entry["r1"], weights):
            if now.numel():
                sums[0] += (now - before).abs().sum().double() * w
                sums[1] += before.abs().sum().double() * w
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(sums)
        num, den = sums.tolist()
        rel = num / max(den, 1e-30)
        entry["last_rel"] = rel
        return rel < spec["threshold"], entry

    def store(self, spec, r1, residual):
        import torch
        import torch.distributed as dist
        tensors = [t.detach() for t in r1] + [t.detach() for t in residual]
        size = sum(t.numel() * t.element_size() for t in tensors)
        self.entries.pop(spec["key"], None)
        limit = spec.get("max_bytes")
        if limit is not None:
            # Набор записей обязан совпадать на всех rank (decide делает all-reduce только при наличии записи),
            # а локальные строки у rank разные: размер берётся как максимум по rank — вытеснение одинаковое.
            if dist.is_initialized() and dist.get_world_size() > 1:
                value = torch.tensor([float(size)], dtype=torch.float64, device=tensors[0].device)
                dist.all_reduce(value, op=dist.ReduceOp.MAX)
                size = int(value.item())
            while self.entries and self.bytes() + size > limit:
                self.entries.pop(next(iter(self.entries)))
                self.stats["evicted"] += 1
            if size > limit:
                return
        self.entries[spec["key"]] = dict(r1=tensors[:len(r1)], residual=tensors[len(r1):], skips=0, bytes=size)
        self.stats["computed"] += 1

    def skip(self, entry):
        entry["skips"] += 1
        self.stats["skipped"] += 1
        return entry["residual"]


def nag_combine(positive, negative, scale, tau, alpha):
    """NAG (ChenDarYen/NAG): экстраполяция, L1-нормализация с порогом tau, смешивание alpha. [.., C] FP32."""
    import torch
    positive, negative = positive.float(), negative.float()
    guidance = positive * scale - negative * (scale - 1.)
    norm_positive = positive.abs().sum(-1, keepdim=True)
    norm_guidance = guidance.abs().sum(-1, keepdim=True)
    ratio = torch.nan_to_num(norm_guidance / norm_positive, nan=10., posinf=10., neginf=10.)
    clipped = guidance / (norm_guidance + 1e-7) * norm_positive * tau
    guidance = torch.where(ratio > tau, clipped, guidance)
    return guidance * alpha + positive * (1. - alpha)


def nag_rows(cond_or_uncond, batch):
    """Индексы строк batch положительной (cond) ветки по transformer_options['cond_or_uncond']."""
    groups = list(cond_or_uncond or [])
    if not groups or batch % len(groups):
        return list(range(batch))
    per = batch // len(groups)
    return [g * per + i for g, c in enumerate(groups) if c == 0 for i in range(per)]


def riflex_frequency_index(head_dim, train_latent_frames, theta=10000.):
    """1-based индекс временной частоты Wan, период которой ближе всего к длине обучения."""
    d = head_dim
    temporal = d - 4 * (d // 6)
    best, best_err = 1, float("inf")
    for j in range(temporal // 2):
        omega = theta ** (-(2. * j) / temporal)
        err = abs(2 * math.pi / omega - train_latent_frames)
        if err < best_err:
            best, best_err = j + 1, err
    return best


def riflex_embedder(embedder, k, latent_frames):
    """Обёртка EmbedND: колонка k-1 временной оси = вращение pos * 0.9*2π/L."""
    import torch
    original = embedder.forward
    omega = 0.9 * 2 * math.pi / max(1, latent_frames)

    def forward(ids):
        out = original(ids)  # [B,1,L,D/2,2,2]
        pos = ids[..., 0].float()
        angle = pos * omega
        cos, sin = torch.cos(angle), torch.sin(angle)
        rot = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(*pos.shape, 2, 2).to(out.dtype)
        out = out.clone()
        out[:, 0, :, k - 1] = rot
        return out
    return original, forward
