"""Параллелизм image-моделей в workers: sequence parallel (токены картинки по GPU) и batch parallel.

Sequence parallel без переписывания моделей: forward каждого блока DiT подменяется обёрткой, которая
1) режет на свою часть только сегмент картинки (текст/референсы остаются целиком),
2) так же режет позиционные эмбеддинги и пересчитывает индексы модуляции (image_layout.Layout),
3) ставит в transformer_options блока ``optimized_attention_override`` — внутри attention ключи и значения
   собираются со всех GPU (all-gather), запросы остаются локальными,
4) после блока собирает выход картинки со всех GPU — снаружи блока модель видит полные тензоры
   (ControlNet, nan_to_num, конкатенации, final layer работают как в обычной ComfyUI).

Семейства: Flux / Flux.2 / Chroma / LongCat (блоки comfy.ldm.flux.layers), Qwen-Image, Lumina 2 / Z-Image
(NextDiT), MOSS-Image. Остальные модели — batch parallel (строки batch по GPU) или только шардирование весов.
"""
import types

import torch
import torch.distributed as dist

from .image_layout import Layout, choose_mode

OVERRIDE_KEY = "optimized_attention_override"


# ------------------------------------------------------------------ коллективы
def _all_gather(out, x):
    """all_gather_into_tensor (NCCL); для gloo (CPU-тесты) — списочный all_gather."""
    if dist.get_backend() == "nccl":
        dist.all_gather_into_tensor(out, x)
        return
    dist.all_gather(list(out.chunk(dist.get_world_size())), x)  # views в out: пишутся на месте


def gather_split(part, layout, dim):
    """Части split-сегмента со всех rank (разной длины) -> полный сегмент по dim."""
    x = part.movedim(dim, 0).contiguous()
    pad = layout.chunk - x.shape[0]
    if pad:
        x = torch.cat([x, x.new_zeros((pad,) + tuple(x.shape[1:]))])
    out = x.new_empty((layout.world * layout.chunk,) + tuple(x.shape[1:]))
    _all_gather(out, x)
    pieces = [out[r * layout.chunk: r * layout.chunk + layout.sizes[r]] for r in range(layout.world)]
    return torch.cat(pieces).movedim(0, dim)


def assemble_full(local, layout, dim):
    """Локальная последовательность (раскладка layout) -> полная, в исходном порядке."""
    s = layout.split_start
    n = layout.b - layout.a
    pre = local.narrow(dim, 0, s)
    mid = gather_split(local.narrow(dim, s, n), layout, dim)
    post = local.narrow(dim, s + n, local.shape[dim] - s - n)
    return torch.cat([pre, mid, post], dim=dim)


def localize(full, layout, dim):
    """Полная последовательность -> локальные строки rank (по глобальным интервалам)."""
    return torch.cat([full.narrow(dim, s, e - s) for s, e in layout.local_ranges()], dim=dim)


def _seq_dim(t, length, preferred):
    if t.shape[preferred] == length:
        return preferred
    raise ValueError(f"PowerShard sequence parallel: тензор {tuple(t.shape)} не имеет длины {length} по оси {preferred}")


# ------------------------------------------------------------------ attention
class SequenceAttention:
    """optimized_attention_override: локальные запросы, ключи/значения со всех GPU."""

    def __init__(self, layout, stats):
        self.layout, self.stats = layout, stats

    def _keys(self, t, dim):
        lay = self.layout
        extra = t.shape[dim] - lay.local_len
        if extra < 0:
            raise ValueError(f"PowerShard sequence parallel: в attention {t.shape[dim]} ключей, а локальных строк "
                             f"{lay.local_len} — блок считает attention не по общей последовательности")
        full = assemble_full(t.narrow(dim, 0, lay.local_len), lay, dim)
        if extra:  # хвост, добавленный внутри attention (MOSS ref_kv) — одинаков на всех rank
            full = torch.cat([full, t.narrow(dim, lay.local_len, extra)], dim=dim)
        return full

    def _mask(self, mask, extra):
        """Маска ключей в локальной раскладке -> полная; строки запросов полной маски -> локальные."""
        if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
            return mask
        lay = self.layout
        if mask.shape[-1] == lay.local_len + extra:  # маска построена внутри блока по локальной длине (Qwen)
            head = assemble_full(mask.narrow(-1, 0, lay.local_len), lay, mask.ndim - 1)
            mask = torch.cat([head, mask.narrow(-1, lay.local_len, extra)], dim=-1) if extra else head
        if mask.shape[-2] == lay.total:
            mask = localize(mask, lay, mask.ndim - 2)
        return mask

    def __call__(self, func, *args, **kwargs):
        q, k, v, heads = args[:4]
        rest = list(args[4:])
        skip = kwargs.get("skip_reshape", False)
        dim = 2 if skip else 1
        if q.shape[dim] != self.layout.local_len:
            raise ValueError(f"PowerShard sequence parallel: {q.shape[dim]} запросов вместо {self.layout.local_len}")
        extra = k.shape[dim] - self.layout.local_len
        k, v = self._keys(k, dim), self._keys(v, dim)
        if rest:
            rest[0] = self._mask(rest[0], extra)
        if "mask" in kwargs:
            kwargs["mask"] = self._mask(kwargs["mask"], extra)
        self.stats["attention_calls"] = self.stats.get("attention_calls", 0) + 1
        return func(q, k, v, heads, *rest, **kwargs)


def _options(transformer_options, layout, stats):
    options = dict(transformer_options or {})
    options[OVERRIDE_KEY] = SequenceAttention(layout, stats)
    return options


# ------------------------------------------------------------------ контекст
class ParallelContext:
    """Состояние на один forward: режим, world/rank, статистика; обёртки блоков читают его."""

    def __init__(self, world, rank):
        self.world, self.rank = world, rank
        self.active = False
        self.families = []
        self.blocks = 0
        self.image_tokens = None     # Lumina / MOSS: число токенов картинки текущего forward
        self.min_tokens = 64
        self.stats = {}

    @property
    def supported(self):
        return self.blocks > 0

    def layout(self, segments):
        lay = Layout(segments, self.world, self.rank)
        if lay.split_len < self.world * self.min_tokens:
            return None
        return lay


def _patch(block, wrapper, ctx):
    original = block.forward

    def forward(*args, **kwargs):
        if not ctx.active:
            return original(*args, **kwargs)
        return wrapper(original, ctx, *args, **kwargs)
    block.forward = forward
    ctx.blocks += 1


# ------------------------------------------------------------------ Flux / Chroma
def _remap_img_relative(lay, dims, prefix):
    if not dims:
        return dims
    return [(lay.local_index(prefix + d[0]) - prefix, lay.local_index(prefix + d[1]) - prefix) + tuple(d[2:])
            for d in dims if lay.local_index(prefix + d[1]) > lay.local_index(prefix + d[0])]


def flux_double(original, ctx, *args, **kwargs):
    names = ("img", "txt", "vec", "pe", "attn_mask")
    values = dict(zip(names, args))
    values.update(kwargs)
    img, txt, pe = values.get("img"), values.get("txt"), values.get("pe")
    if pe is None or img is None or txt is None:
        return original(*args, **kwargs)
    prefix = txt.shape[1]
    lay = ctx.layout([(prefix, False), (img.shape[1], True)])
    if lay is None:
        return original(*args, **kwargs)
    values["img"] = img[:, lay.a - prefix: lay.b - prefix]
    values["pe"] = localize(pe, lay, _seq_dim(pe, lay.total, 2))
    if values.get("modulation_dims_img"):
        values["modulation_dims_img"] = _remap_img_relative(lay, values["modulation_dims_img"], prefix)
    values["transformer_options"] = _options(values.get("transformer_options"), lay, ctx.stats)
    out_img, out_txt = original(**values)
    return gather_split(out_img, lay, 1), out_txt


def flux_single(original, ctx, *args, **kwargs):
    names = ("img", "vec", "pe", "attn_mask")
    values = dict(zip(names, args))
    values.update(kwargs)
    if "x" in values and "img" not in values:
        values["img"] = values.pop("x")
    x, pe = values.get("img"), values.get("pe")
    options = values.get("transformer_options") or {}
    window = options.get("img_slice")
    if pe is None or x is None or not window:
        return original(*args, **kwargs)
    prefix, total = int(window[0]), int(window[1])
    if x.shape[1] != total:
        return original(*args, **kwargs)
    lay = ctx.layout([(prefix, False), (total - prefix, True)])
    if lay is None:
        return original(*args, **kwargs)
    values["img"] = localize(x, lay, 1)
    values["pe"] = localize(pe, lay, _seq_dim(pe, total, 2))
    if values.get("modulation_dims"):
        values["modulation_dims"] = lay.remap_ranges(values["modulation_dims"])
    values["transformer_options"] = _options(options, lay, ctx.stats)
    img = values.pop("img")
    out = original(img, **values)
    return assemble_full(out, lay, 1)


# ------------------------------------------------------------------ Qwen-Image
def qwen_block(original, ctx, *args, **kwargs):
    names = ("hidden_states", "encoder_hidden_states", "encoder_hidden_states_mask", "temb", "image_rotary_emb",
             "timestep_zero_index", "transformer_options")
    values = dict(zip(names, args))
    values.update(kwargs)
    img, txt, pe = values.get("hidden_states"), values.get("encoder_hidden_states"), values.get("image_rotary_emb")
    if img is None or txt is None or pe is None:
        return original(*args, **kwargs)
    prefix = txt.shape[1]
    lay = ctx.layout([(prefix, False), (img.shape[1], True)])
    if lay is None:
        return original(*args, **kwargs)
    values["hidden_states"] = img[:, lay.a - prefix: lay.b - prefix]
    values["image_rotary_emb"] = localize(pe, lay, _seq_dim(pe, lay.total, 2))
    zero = values.get("timestep_zero_index")
    if zero is not None:
        values["timestep_zero_index"] = lay.local_index(prefix + int(zero)) - prefix
    values["transformer_options"] = _options(values.get("transformer_options"), lay, ctx.stats)
    out_txt, out_img = original(**values)
    return out_txt, gather_split(out_img, lay, 1)


# ------------------------------------------------------------------ Lumina 2 / Z-Image
def lumina_layer(original, ctx, *args, **kwargs):
    names = ("x", "x_mask", "freqs_cis", "adaln_input")
    values = dict(zip(names, args))
    values.update(kwargs)
    x, freqs = values.get("x"), values.get("freqs_cis")
    n = ctx.image_tokens
    if x is None or freqs is None or not n or n >= x.shape[1]:
        return original(*args, **kwargs)
    prefix = x.shape[1] - n
    lay = ctx.layout([(prefix, False), (n, True)])
    if lay is None:
        return original(*args, **kwargs)
    values["x"] = localize(x, lay, 1)
    values["freqs_cis"] = localize(freqs, lay, _seq_dim(freqs, lay.total, 1))
    if values.get("timestep_zero_index"):
        values["timestep_zero_index"] = lay.remap_ranges(values["timestep_zero_index"])
    mask = values.get("x_mask")
    if isinstance(mask, torch.Tensor) and mask.ndim >= 2 and mask.shape[-2] == lay.total:
        values["x_mask"] = localize(mask, lay, mask.ndim - 2)
    values["transformer_options"] = _options(values.get("transformer_options"), lay, ctx.stats)
    out = original(values.pop("x"), values.pop("x_mask"), values.pop("freqs_cis"), values.pop("adaln_input", None),
                   **values)
    return assemble_full(out, lay, 1)


def _capture_lumina_tokens(model, ctx):
    original = model.patchify_and_embed

    def patchify_and_embed(*args, **kwargs):
        out = original(*args, **kwargs)
        full, cap_len = out[0], out[3]
        ctx.image_tokens = int(full.shape[1] - (cap_len[0] if isinstance(cap_len, (list, tuple)) else cap_len))
        return out
    model.patchify_and_embed = patchify_and_embed


# ------------------------------------------------------------------ MOSS-Image
def moss_block(original, ctx, *args, **kwargs):
    names = ("x", "mod", "pe", "transformer_options", "ref", "ref_kv", "kv_out", "mask")
    values = dict(zip(names, args))
    values.update(kwargs)
    x, pe, n = values.get("x"), values.get("pe"), ctx.image_tokens
    if x is None or pe is None or not n or values.get("kv_out") is not None or n > x.shape[1]:
        return original(*args, **kwargs)
    lay = ctx.layout([(n, True), (x.shape[1] - n, False)] if x.shape[1] > n else [(n, True)])
    if lay is None:
        return original(*args, **kwargs)
    values["x"] = localize(x, lay, 1)
    values["pe"] = localize(pe, lay, _seq_dim(pe, lay.total, 1))
    ref = values.get("ref")
    if ref is not None:
        values["ref"] = (lay.local_index(int(ref[0])),) + tuple(ref[1:])
    values["transformer_options"] = _options(values.get("transformer_options"), lay, ctx.stats)
    out = original(values.pop("x"), values.pop("mod"), values.pop("pe"), values.pop("transformer_options"),
                   values.pop("ref", None), values.pop("ref_kv", None), values.pop("kv_out", None),
                   mask=values.pop("mask", None))
    return assemble_full(out, lay, 1)


def _capture_moss_tokens(model, ctx):
    def hook(module, args, kwargs):
        x = args[0] if args else kwargs.get("x")
        if isinstance(x, torch.Tensor) and x.ndim == 4:
            ctx.image_tokens = int(x.shape[-2] * x.shape[-1])
    model.register_forward_pre_hook(hook, with_kwargs=True)


# ------------------------------------------------------------------ установка
def install_sequence_parallel(model, ctx):
    """Найти известные блоки в модели и подменить их forward. Возвращает список семейств."""
    families = []
    try:
        from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
        doubles = [m for m in model.modules() if isinstance(m, DoubleStreamBlock)]
        singles = [m for m in model.modules() if isinstance(m, SingleStreamBlock)]
        if doubles or singles:
            for block in doubles:
                _patch(block, flux_double, ctx)
            for block in singles:
                _patch(block, flux_single, ctx)
            families.append("flux")
    except ImportError:
        pass
    name = type(model).__name__
    module = type(model).__module__
    if name == "QwenImageTransformer2DModel" and module.endswith("qwen_image.model"):
        for block in model.transformer_blocks:
            _patch(block, qwen_block, ctx)
        families.append("qwen_image")
    elif module.endswith("lumina.model") and hasattr(model, "layers") and hasattr(model, "patchify_and_embed"):
        for block in model.layers:
            _patch(block, lumina_layer, ctx)
        _capture_lumina_tokens(model, ctx)
        families.append("lumina2")
    elif name == "MOSSImage":
        for block in model.layers:
            _patch(block, moss_block, ctx)
        _capture_moss_tokens(model, ctx)
        families.append("moss_image")
    ctx.families = families
    return families


# ------------------------------------------------------------------ batch parallel
def _rows(n, world, rank):
    from .image_layout import split_sizes
    sizes = split_sizes(n, world)
    start = sum(sizes[:rank])
    return sizes, start, sizes[rank]


def split_batch(value, batch, start, count):
    """Тензоры с первой осью batch -> строки [start, start+count) (пустой rank берёт строку 0)."""
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == batch:
            return value[start:start + count] if count else value[:1]
        return value
    if isinstance(value, dict):
        return {k: split_batch(v, batch, start, count) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(split_batch(v, batch, start, count) for v in value)
    return value


def gather_batch(out, count, sizes, world):
    """Выходы строк всех rank -> полный batch. Rank без строк считал строку-заглушку: её не отдаём."""
    chunk = max(sizes)
    send = out[:count] if count else out[:0]
    if send.shape[0] < chunk:
        filler = out[:1].expand((chunk - send.shape[0],) + tuple(out.shape[1:]))
        send = torch.cat([send, filler])
    send = send.contiguous()
    full = send.new_empty((world * chunk,) + tuple(send.shape[1:]))
    _all_gather(full, send)
    return torch.cat([full[r * chunk: r * chunk + sizes[r]] for r in range(world)])


def image_tokens_estimate(x, families=()):
    if not isinstance(x, torch.Tensor) or x.ndim < 4:
        return 0
    spatial = 1
    for s in x.shape[2:]:
        spatial *= int(s)
    return spatial if "moss_image" in families else spatial // 4  # MOSS: патч 1, остальные DiT: 2x2


def run_parallel(call, ctx, requested, x, args, kwargs):
    """Выбрать режим (одинаково на всех rank — входы одинаковые) и выполнить forward."""
    world, rank = ctx.world, ctx.rank
    batch = int(x.shape[0]) if isinstance(x, torch.Tensor) and x.ndim else 1
    mode, reason = choose_mode(requested, ctx.supported, batch, world, image_tokens_estimate(x, ctx.families),
                               ctx.min_tokens)
    ctx.stats = dict(mode=mode, reason=reason)
    if mode == "batch":
        sizes, start, count = _rows(batch, world, rank)
        local_args = split_batch(args, batch, start, count)
        options = dict(kwargs.get("transformer_options") or {})
        local_kwargs = split_batch({k: v for k, v in kwargs.items() if k != "transformer_options"}, batch, start, count)
        if "sigmas" in options:  # sample_sigmas (расписание) не делится, даже если его длина совпала с batch
            options["sigmas"] = split_batch(options["sigmas"], batch, start, count)
        local_kwargs["transformer_options"] = options
        out = call(*local_args, **local_kwargs)
        return gather_batch(out, count, sizes, world), ctx.stats
    ctx.active = mode == "sequence"
    try:
        return call(*args, **kwargs), ctx.stats
    finally:
        ctx.active = False
        ctx.image_tokens = None


def install(model, ctx):
    return install_sequence_parallel(model, ctx)


__all__ = ["ParallelContext", "install", "run_parallel", "gather_split", "assemble_full", "localize",
           "SequenceAttention", "types"]
