from dataclasses import asdict, dataclass
import re
import warnings
import math


@dataclass(frozen=True)
class DistributedConfig:
    gpu_ids: tuple[str, ...] = ("all",)
    backend: str = "fsdp2_sequence"
    precision: str = "fp16"
    attention: str = "exact_chunked"
    reserve_gib: float = 2.0
    timeout_s: int | None = None  # ignored legacy field; no RPC deadline
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
    memory_policy: str = "auto"
    weight_placement: str = "gpu"
    sequence_mode: str = "token"
    # Baseline ~45s used Safe FP32 wire. Normalized FP16 adds one MAX
    # collective per attention block; make the tradeoff an explicit opt-in.
    sequence_comm_dtype: str = "fp32"
    allow_host_wrappers: bool = False

    def __post_init__(self):
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
        if self.backend == "fsdp2":
            warnings.warn("Legacy backend=fsdp2 заменён на fsdp2_sequence", stacklevel=2)
            object.__setattr__(self, "backend", "fsdp2_sequence")
        if self.backend != "fsdp2_sequence":
            raise ValueError("Поддерживается fsdp2_sequence")
        if self.precision not in ("fp16", "int8_fp16"):
            raise ValueError("precision: fp16 или int8_fp16")
        if self.requested_attention not in ("auto", "sdpa", "flash_attn", "vllm_flash_attn", "sageattention", "math"):
            raise ValueError("Неизвестный attention backend")
        if not math.isfinite(self.reserve_gib) or self.reserve_gib < 0:
            raise ValueError("Некорректный reserve_gib")
        object.__setattr__(self, "timeout_s", None)
        if min(self.query_chunk, self.key_chunk, self.dequant_rows) <= 0:
            raise ValueError("Размеры chunks должны быть положительными")
        if self.prefetch_blocks not in (0, 1, 2):
            raise ValueError("prefetch_blocks: 0, 1 или 2")
        if self.numa_policy not in ("none", "auto", "bind"):
            raise ValueError("numa_policy: none / auto (CPU affinity) / bind (CPU+memory)")
        if self.memory_policy not in ("manual", "auto"):
            raise ValueError("memory_policy: manual / auto")
        if self.weight_placement not in ("gpu", "cpu", "ats"):
            raise ValueError("weight_placement: gpu / cpu / ats")
        # ATS uses CUDA managed shards, NEVER CPUOffloadPolicy.
        if self.weight_placement == "gpu" and self.cpu_offload:
            object.__setattr__(self, "weight_placement", "cpu")
        object.__setattr__(self, "cpu_offload", self.weight_placement == "cpu")
        if self.sequence_mode not in ("token", "ulysses"):
            raise ValueError("sequence_mode: token / ulysses")
        if self.sequence_comm_dtype not in ("fp16", "fp32"):
            raise ValueError("sequence_comm_dtype: fp16 / fp32")
        if self.allow_host_wrappers:
            warnings.warn("allow_host_wrappers=True: сторонние callbacks/wrappers будут выполняться на HOST "
                          "до/после RPC в workers. Wrappers, меняющие tensors (LoRA-стиль), дадут неверный результат "
                          "или NaN — PowerShard не может их проверить.")

    def to_dict(self):
        value = asdict(self)
        value.pop("timeout_s")
        value.pop("allow_unverified")
        return value

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
            "communication_dtype": ("parameter dtype per FSDP group; normalized QKV/output "
                                    + self.sequence_comm_dtype + "; final residual FP32; Safe math exchange FP32"),
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
