# Переработано из Veda-on-ComfyUI (MIT, (c) 2026 veda-sparse), veda_comfy/settings.py. См. NOTICE.md.
"""Разбор настроек ноды Veda без torch (host). Бюджет — dict(ratio=...) или dict(tiles=...)."""
import re

_SPLIT = re.compile(r"[\s,;，；、]+")
_RANGE = re.compile(r"^(\d+)\s*[-~–—]\s*(\d+)$")
_PERCENT = re.compile(r"^(\d+(?:\.\d+)?)\s*[%％]$")
_TILES = re.compile(r"^(\d+)$")
TRAINED = "trained"
DEFAULT_BUDGET = "32"   # как в официальной ноде: абсолютное число тайлов не «схлопывается» на малых сетках
BACKENDS = ("auto", "triton-int8", "triton-fp16", "torch-sdpa")


def parse_index_list(text, what):
    """'0, 1, 47-49' -> [0, 1, 47, 48, 49] (с нуля, диапазоны включительно)."""
    indices = set()
    text = re.sub(r"\s*([-~–—])\s*", r"\1", (text or "").strip())
    for token in _SPLIT.split(text):
        if not token:
            continue
        if token.isdigit():
            indices.add(int(token))
            continue
        match = _RANGE.match(token)
        if match is None:
            raise ValueError(f'{what}: не понимаю {token!r}. Пишите индексы с нуля и диапазоны, например "0, 1, 47-49"')
        lo, hi = sorted((int(match.group(1)), int(match.group(2))))
        indices.update(range(lo, hi + 1))
    return sorted(indices)


def parse_sparsity(text, what, trained=None):
    """'90%' -> пропустить 90% тайлов; '24' -> оставить 24 тайла; 'trained' -> бюджет обучения предиктора."""
    value = (text or "").strip()
    if value.lower() == TRAINED:
        if trained is None:
            raise ValueError(f"{what}: 'trained' требует предиктор с бюджетом обучения")
        return dict(trained)
    match = _PERCENT.match(value)
    if match is not None:
        percent = float(match.group(1))
        if not 0.0 <= percent < 100.0:
            raise ValueError(f'{what}: {value} — нужен процент меньше 100, например "90%"')
        return dict(ratio=1.0 - percent / 100.0)
    match = _TILES.match(value)
    if match is not None and int(match.group(1)) > 0:
        return dict(tiles=float(match.group(1)))
    raise ValueError(f'{what}: не понимаю {value!r}. Пишите "trained", процент разреженности ("90%") или '
                     'целое число тайлов на тайл запроса ("32")')


def keeps_all(budget):
    return "ratio" in budget and budget["ratio"] >= 1.0


def format_budget(budget):
    if "tiles" in budget:
        return f"{budget['tiles']:g} тайлов"
    return f"{100.0 * (1.0 - min(budget['ratio'], 1.0)):g}%"


def step_index(sigma, sample_sigmas):
    """Шаг sampler (с нуля), в интервал которого попадает sigma (как в официальной ноде)."""
    sigmas = [float(s) for s in sample_sigmas]
    if len(sigmas) < 2:
        return None
    tolerance = 1e-5 * max(1.0, abs(sigma))
    for i in range(len(sigmas) - 1):
        high, low = sigmas[i], sigmas[i + 1]
        if high + tolerance >= sigma > low + tolerance:
            return i
    if sigma >= sigmas[0]:
        return 0
    return len(sigmas) - 2 if sigma >= sigmas[-1] - tolerance else None
