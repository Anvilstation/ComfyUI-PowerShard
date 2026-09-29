from dataclasses import asdict, dataclass
import re
import warnings
import math


@dataclass(frozen=True)
class DistributedConfig:
    gpu_ids: tuple[str, ...] = ("0", "1", "2")
    backend: str = "fsdp2"
    precision: str = "fp16"
    attention: str = "exact_chunked"
    reserve_gib: float = 2.0
    timeout_s: int = 600
    query_chunk: int = 128
    key_chunk: int = 512
    dequant_rows: int = 256
    allow_unverified: bool = False
    release_after_sampling: bool = True
    cpu_offload: bool = False
    pin_memory: bool = True
    prefetch_blocks: int = 0
    numa_policy: str = "none"
    attention_backend: str | None = None
    allow_fallback: bool = True
    memory_policy: str = "manual"
    weight_placement: str = "gpu"
    sequence_mode: str = "token"
    sequence_comm_dtype: str = "fp32"
    memory_profile: str = "custom"
    workspace_mib: int = 256
    stage_cache_mib: int = 64

    def __post_init__(self):
        if self.memory_profile not in ("custom", "ram_min"):
            raise ValueError("memory_profile: custom / ram_min")
        if not isinstance(self.workspace_mib, int) or self.workspace_mib < 1:
            raise ValueError("workspace_mib должен быть положительным целым")
        if not isinstance(self.stage_cache_mib, int) or self.stage_cache_mib < 0:
            raise ValueError("stage_cache_mib должен быть целым >= 0")
        ids = self.gpu_ids.split(",") if isinstance(self.gpu_ids, str) else self.gpu_ids
        ids = tuple(str(x).strip() for x in ids)
        if not ids or any(not x for x in ids):
            raise ValueError("Выберите непустой набор GPU")
        if "all" in ids and ids != ("all",):
            raise ValueError("all нельзя смешивать с отдельными GPU IDs")
        if len(set(ids)) != len(ids):
            warnings.warn("Повторные GPU IDs удалены с сохранением порядка", stacklevel=2)
        object.__setattr__(self, "gpu_ids", tuple(dict.fromkeys(ids)))
        if not all(re.fullmatch(r"\d+|(?:GPU|MIG)-[0-9a-fA-F-]+|all", x) for x in self.gpu_ids):
            raise ValueError("Некорректный идентификатор GPU")
        if self.backend not in ("fsdp2", "fsdp2_sequence"):
            raise ValueError("Поддерживаются fsdp2 и fsdp2_sequence")
        if self.precision not in ("fp16", "int8_fp16"):
            raise ValueError("precision: fp16 или int8_fp16")
        if self.requested_attention not in ("auto", "sdpa", "flash_attn", "vllm_flash_attn", "sageattention", "math"):
            raise ValueError("Неизвестный attention backend")
        if not math.isfinite(self.reserve_gib) or not math.isfinite(self.timeout_s) or not 0 <= self.reserve_gib or not 0 < self.timeout_s:
            raise ValueError("Некорректные reserve_gib / timeout_s")
        if min(self.query_chunk, self.key_chunk, self.dequant_rows) <= 0:
            raise ValueError("Размеры chunks должны быть положительными")
        if self.prefetch_blocks not in (0, 1, 2):
            raise ValueError("prefetch_blocks: 0, 1 или 2")
        if self.numa_policy not in ("none", "auto", "bind"):
            raise ValueError("numa_policy: none / auto (CPU affinity) / bind (CPU+memory)")
        if self.memory_policy not in ("manual", "auto"):
            raise ValueError("memory_policy: manual / auto")
        if self.min_vram:
            object.__setattr__(self, "cpu_offload", True)
            object.__setattr__(self, "prefetch_blocks", 0)
            object.__setattr__(self, "memory_policy", "auto")
        if self.weight_placement not in ("gpu", "cpu", "ats"):
            raise ValueError("weight_placement: gpu / cpu / ats")
        if self.weight_placement in ("cpu", "ats") and not self.cpu_offload:
            # ats — совместимый UI alias CPUOffloadPolicy + диагностика, не ATS allocation.
            object.__setattr__(self, "cpu_offload", True)
        if self.sequence_mode not in ("token", "ulysses"):
            raise ValueError("sequence_mode: token / ulysses (требует heads % world == 0)")
        if self.sequence_comm_dtype not in ("fp16", "fp32"):
            raise ValueError("sequence_comm_dtype: fp16 / fp32")
        if self.sequence_mode == "ulysses" and self.backend != "fsdp2_sequence":
            warnings.warn("sequence_mode=ulysses действует только при backend=fsdp2_sequence; при fsdp2 он не используется")

    def to_dict(self):
        return asdict(self)

    @property
    def min_vram(self):
        return self.memory_profile == "ram_min"

    def memory_settings(self):
        return dict(profile=self.memory_profile, parameter_residency="cpu_shards" if self.cpu_offload else "gpu_shards",
            fsdp_granularity="attention_mlp" if self.min_vram else "block", prefetch_blocks=self.prefetch_blocks,
            workspace_limit_bytes=self.workspace_mib*2**20 if self.min_vram else None,
            prepared_weights=not self.min_vram, linear_output_rows=self.dequant_rows if self.min_vram else None,
            conditioning_cache_device="cpu" if self.min_vram else "worker_device",
            conditioning_cache_limit_bytes=self.stage_cache_mib*2**20 if self.min_vram else None,
            sdpa_execution="query_head_tiles" if self.min_vram else "automatic_dispatch",
            note="Workspace — цель планирования, не общий лимит VRAM. Активные веса/activations/output/NCCL остаются на GPU.")

    @property
    def requested_attention(self):
        # Отсутствующее новое поле сохраняет математику старых API/workflows.
        return self.attention_backend or ("math" if self.attention == "exact_chunked" else self.attention)

    def numeric_policy(self, patch=None):
        """Не смешивать компактный UI preset с пятью независимыми свойствами backend."""
        return {
            "storage_dtype": "int8 + float16/float32 auxiliary" if self.precision == "int8_fp16" else "float16 + float32 islands",
            "compute_dtype": "FP16 scaled GEMM; FP32 condition/norm/SiLU/softmax/final" if patch and patch.active else "legacy FP16; FP32 native islands",
            "residual_dtype": "float32" if patch and patch.active else "float16",
            "activation_dtype": "FP32 restored; bounded FP16 GEMM inputs" if patch and patch.active else "legacy",
            "communication_dtype": f"parameter dtype per FSDP group; H3 token KV {self.sequence_comm_dtype} (scaled if FP16); Ulysses QKV native dtype; Qwen KV/hidden FP32",
            "weight_placement_requested": self.weight_placement,
            "weight_placement_effective": "CPUOffloadPolicy" if self.cpu_offload else "GPU shards",
            "memory_settings": self.memory_settings(),
            "quantization_format": "int8_tensorwise_convrot" if self.precision == "int8_fp16" else "none; BF16 source converted locally",
            "attention_backend": self.requested_attention,
            "distributed_backend": self.backend,
        }


def shard_bounds(rows: int, rank: int, world_size: int):
    """Разбиение Shard(0) / torch.chunk, включая пустые последние ранги."""
    if rows < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Некорректные границы шарда")
    width = (rows + world_size - 1) // world_size
    return min(rows, rank * width), min(rows, (rank + 1) * width)
