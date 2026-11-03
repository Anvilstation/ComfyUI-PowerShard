# Переработано из Veda-on-ComfyUI (MIT, (c) 2026 veda-sparse), veda_comfy/core/bundle.py. См. NOTICE.md.
"""Файл предиктора Veda: проекции [H, 3D, D] на слой + планы тайлов (формат miowtion-veda-predictor-v1).

Часть без torch (read_header_metadata / describe_metadata) нужна host-ноде: проверить файл и геометрию H3
до запуска workers. Загрузка тензоров — только в workers (load_bundle).
"""
import dataclasses
import json
import os
import struct

FORMAT = "miowtion-veda-predictor-v1"
_SCALE_SUFFIX = ".__scale"
_STORAGE = ("float32", "bfloat16", "float8_e4m3fn")


class BundleError(ValueError):
    """Файл не является пригодным предиктором Veda (сообщение — для пользователя)."""


def read_header_metadata(path):
    """__metadata__ safetensors без torch."""
    name = os.path.basename(path)
    try:
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if not 2 <= size <= 64 * 1024 * 1024:
                raise ValueError("недопустимый размер заголовка")
            header = json.loads(f.read(size))
    except (OSError, ValueError, struct.error) as error:
        raise BundleError(f"{name}: не читается как safetensors ({error})") from error
    metadata = dict(header.get("__metadata__") or {})
    if metadata.get("format") != FORMAT:
        raise BundleError(f"{name} — не предиктор Veda (format {metadata.get('format')!r}, нужен {FORMAT!r}). "
                          "В models/veda кладите только файлы предиктора Veda")
    return metadata


def geometry(metadata):
    """(num_layers, num_heads, head_dim) из метаданных."""
    try:
        return int(metadata["num_layers"]), int(metadata["num_heads"]), int(metadata["head_dim"])
    except (KeyError, ValueError) as error:
        raise BundleError(f"неполные метаданные предиктора ({error})") from error


def trained_budgets(metadata):
    """Бюджеты обучения (generated, reference) как dict(kind, value) — без torch."""
    keep = metadata.get("keep_ratio")
    default = dict(ratio=float(keep)) if keep is not None else None

    def one(prefix, fallback):
        kind = metadata.get(f"{prefix}_budget_kind")
        if kind is None:
            if fallback is None:
                raise BundleError(f"в метаданных нет {prefix}_budget_kind")
            return fallback
        try:
            value = float(metadata[f"{prefix}_budget_value"])
        except (KeyError, ValueError) as error:
            raise BundleError(f"{prefix}_budget_value: {error}") from error
        if kind not in ("ratio", "tiles"):
            raise BundleError(f"{prefix}_budget_kind: неизвестный вид {kind!r}")
        return {kind: value}
    generated = one("target", default)
    return generated, one("ref", generated)


@dataclasses.dataclass
class PredictorBundle:
    path: str
    num_layers: int
    num_heads: int
    head_dim: int
    generated: object
    reference: object
    plans: object
    proj_q: list
    proj_k: list
    metadata: dict

    def describe(self):
        return (f"{os.path.basename(self.path)} · {self.num_layers} слоёв x {self.num_heads} голов · "
                f"шаг обучения {self.metadata.get('step', '?')} · планы: {self.plans.summary()}")


def load_bundle(path):
    """Строгая загрузка: проекции хранятся на host в bf16 (fp8-файлы деквантуются по per-head scale)."""
    import torch
    from safetensors import safe_open
    from . import plans as veda_plans
    from . import selection
    name = os.path.basename(path)
    metadata = read_header_metadata(path)
    num_layers, num_heads, head_dim = geometry(metadata)
    generated, reference = (selection.Budget(**b) for b in trained_budgets(metadata))
    try:
        plan_json = json.loads(metadata["plans"])
    except (KeyError, ValueError) as error:
        raise BundleError(f"{name}: нет планов тайлов ({error})") from error
    stored = metadata.get("dtype", "float32")
    if stored not in _STORAGE:
        raise BundleError(f"{name}: неизвестный dtype хранения {stored!r}")
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    shape = (num_heads, 3 * head_dim, head_dim)
    proj_q, proj_k = [], []
    for layer in range(num_layers):
        for key_name, out in (("proj_q", proj_q), ("proj_k", proj_k)):
            key = f"layers.{layer}.{key_name}"
            value = tensors.pop(key, None)
            if value is None or tuple(value.shape) != shape:
                raise BundleError(f"{name}: {key} отсутствует или не {list(shape)}")
            if stored == "float8_e4m3fn":
                scale = tensors.pop(key + _SCALE_SUFFIX, None)
                if scale is None or tuple(scale.shape) != (num_heads,):
                    raise BundleError(f"{name}: у {key} нет per-head fp8 scale")
                value = value.float() * scale.float().view(-1, 1, 1)
            out.append(value.to(torch.bfloat16).contiguous())
    if tensors:
        raise BundleError(f"{name}: лишние тензоры {sorted(tensors)[:4]}")
    table = veda_plans.PlanTable([veda_plans.TilePlan.from_json(p) for p in plan_json.values()])
    for plan in table.plans.values():
        if plan.num_layers != num_layers or plan.num_heads != num_heads:
            raise BundleError(f"{name}: план {plan.name} {plan.num_layers}x{plan.num_heads}, "
                              f"предиктор {num_layers}x{num_heads}")
    return PredictorBundle(path, num_layers, num_heads, head_dim, generated, reference, table, proj_q, proj_k,
                           metadata)
