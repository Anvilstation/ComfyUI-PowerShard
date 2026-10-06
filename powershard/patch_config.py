"""Настройки patch — только immutable JSON values, не bound methods/closures."""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class H3PatchConfig:
    enabled: bool = False
    fp16_safe: bool = True
    debug_finite: bool = False
    mlp_chunk_tokens: int = 512
    mlp_chunk_mode: str = "off"

    def __post_init__(self):
        if self.mlp_chunk_tokens < 1:
            raise ValueError("mlp_chunk_tokens должен быть положительным; для отключения выберите off")
        if self.mlp_chunk_mode not in ("manual", "auto", "off"):
            raise ValueError("mlp_chunk_mode: auto / manual / off")

    @property
    def active(self):
        return self.enabled and self.fp16_safe

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
