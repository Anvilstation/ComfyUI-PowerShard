"""Раскладка последовательности для sequence parallel image-моделей (без torch).

Последовательность блока DiT — несколько сегментов подряд: текст, картинка, референсы. Делится по GPU только
сегмент картинки («split»); остальные («replicated») целиком есть у каждого rank. Локальная последовательность
rank — те же сегменты в том же порядке, но вместо split-сегмента его часть [a, b). Глобальные позиции
локальных строк идут по возрастанию, поэтому любой глобальный интервал [s, e) переходит в один локальный
интервал — на этом держится пересчёт индексов модуляции (Flux modulation_dims, Qwen/Lumina
timestep_zero_index, MOSS ref) без знания модели.
"""


def split_sizes(n, world):
    """Размеры частей n строк на world rank: первые n % world получают на одну больше."""
    base, extra = divmod(int(n), int(world))
    return [base + (1 if r < extra else 0) for r in range(world)]


class Layout:
    """segments: [(длина, split: bool)], ровно один split-сегмент."""

    def __init__(self, segments, world, rank):
        self.segments = [(int(n), bool(s)) for n, s in segments]
        if sum(1 for _, s in self.segments if s) != 1:
            raise ValueError(f"Нужен ровно один split-сегмент: {segments}")
        self.world, self.rank = int(world), int(rank)
        self.total = sum(n for n, _ in self.segments)
        start = 0
        for n, s in self.segments:
            if s:
                self.split_start, self.split_len = start, n
            start += n
        self.sizes = split_sizes(self.split_len, self.world)
        self.offsets = [sum(self.sizes[:r]) for r in range(self.world)]
        self.a = self.split_start + self.offsets[self.rank]
        self.b = self.a + self.sizes[self.rank]
        self.local_len = self.total - self.split_len + (self.b - self.a)
        self.chunk = max(self.sizes) if self.sizes else 0

    # ------------------------------------------------------------------ позиции
    def local_ranges(self):
        """Глобальные интервалы, из которых состоят локальные строки (по порядку)."""
        out = []
        if self.split_start > 0:
            out.append((0, self.split_start))
        if self.b > self.a:
            out.append((self.a, self.b))
        end = self.split_start + self.split_len
        if end < self.total:
            out.append((end, self.total))
        return out

    def local_positions(self):
        return [p for s, e in self.local_ranges() for p in range(s, e)]

    def local_index(self, g):
        """Число локальных строк с глобальной позицией < g."""
        g = max(0, min(int(g), self.total))
        return sum(max(0, min(e, g) - s) for s, e in self.local_ranges())

    def remap_ranges(self, ranges):
        """[(s, e, *extra)] глобальные -> локальные; пустые интервалы удаляются."""
        out = []
        for item in ranges:
            s, e = self.local_index(item[0]), self.local_index(item[1])
            if e > s:
                out.append((s, e) + tuple(item[2:]))
        return out

    def describe(self):
        return dict(total=self.total, split=[self.split_start, self.split_len], local=[self.a, self.b],
                    local_len=self.local_len, sizes=self.sizes)


def choose_mode(requested, family_supported, batch, world, image_tokens, min_tokens_per_rank=64):
    """auto: sequence, если семейство поддерживает и токенов картинки достаточно; иначе batch (если строк > 1).

    Возвращает (mode, причина). mode: sequence / batch / replicate.
    """
    if world <= 1:
        return "replicate", "один GPU"
    if requested == "replicate":
        return "replicate", "выбрано"
    seq_ok = family_supported and image_tokens >= world * min_tokens_per_rank
    if requested == "sequence":
        if not family_supported:
            return ("batch" if batch > 1 else "replicate"), "sequence parallel для этого семейства не реализован"
        if not seq_ok:
            return ("batch" if batch > 1 else "replicate"), f"мало токенов картинки ({image_tokens}) на {world} GPU"
        return "sequence", "выбрано"
    if requested == "batch":
        return ("batch", "выбрано") if batch > 1 else ("replicate", "batch=1: делить нечего")
    if seq_ok:
        return "sequence", "auto: семейство поддерживает sequence parallel"
    if batch > 1:
        return "batch", "auto: строки batch (cond/uncond, batch_size) по GPU"
    return "replicate", "auto: batch=1 и нет sequence parallel — только шардирование весов"
