"""JSON-only Spectrum policy. Не импортирует CUDA или ComfyUI при регистрации нод."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SpectrumConfig:
    enabled: bool = False
    degree: int = 2
    warmup: int = 3
    tail: int = 1
    max_forecast: int = 1
    history_size: int = 6
    history_mib: int = 512
    history_device: str = "cpu"
    ridge: float = .001
    blend: float = .5
    audio_blend: float = 0.
    allow_any_sampler: bool = False

    def __post_init__(self):
        import math
        if not (1 <= self.degree <= 4 and self.history_size >= self.degree+1):
            raise ValueError("Spectrum: degree 1..4, history_size >= degree+1")
        if self.warmup < self.degree+1 or self.tail < 1 or self.max_forecast < 1 or self.history_mib < 0:
            raise ValueError("Spectrum: warmup >= degree+1, tail/max_forecast >= 1, history_mib >= 0")
        if self.history_device not in ("cpu","cuda","gpu"):
            raise ValueError("Spectrum history_device: cpu или cuda")
        if self.history_device=="gpu":object.__setattr__(self,"history_device","cuda")
        if not math.isfinite(self.ridge) or self.ridge <= 0 or not 0 <= self.blend <= 1 or not 0 <= self.audio_blend <= 1:
            raise ValueError("Spectrum: ridge > 0, blend 0..1")

    def to_dict(self):return asdict(self)
