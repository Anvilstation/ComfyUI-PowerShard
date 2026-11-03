"""LoRA через родной разбор ComfyUI (comfy.lora) для image-моделей и MiniMax H3.

Ключи LoRA сопоставляет сам ComfyUI (model_lora_keys_unet: kohya, diffusers, PEFT, LyCORIS, fused qkv со
смещениями, банки PDD-голов H3), а дельта считается его же адаптерами на нулевом весе:
calculate_weight(0) = 0 + strength * delta. Полный вес модели не нужен — в worker дельта режется на строки
своего shard. Поддерживаются только аддитивные адаптеры (LoRA, LoHa, LoKr, diff); DoRA, OFT, BOFT, GLoRA
зависят от исходного веса и отклоняются явно.
"""
from pathlib import Path

import torch

ADDITIVE = ("LoRAAdapter", "LoHaAdapter", "LoKrAdapter")


def _patch_key(k):
    if isinstance(k, str):
        return k, None, None
    return k[0], k[1], (k[2] if len(k) > 2 else None)


class ComfyLoraDeltas:
    """specs: [{"path", "strength"}]. base_model — BaseModel (можно на meta), prefix — префикс параметров в
    state_dict BaseModel ("diffusion_model.")."""

    def __init__(self, base_model, specs, prefix="diffusion_model."):
        import comfy.lora
        import comfy.utils
        self.prefix = prefix
        self.patches = {}
        self.report = []
        key_map = comfy.lora.model_lora_keys_unet(base_model, {})
        for spec in specs:
            path, strength = str(spec["path"]), float(spec["strength"])
            lora = comfy.utils.load_torch_file(path, safe_load=True)
            loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
            used, rejected = 0, []
            for key, adapter in loaded.items():
                name, offset, function = _patch_key(key)
                kind = type(adapter).__name__
                if isinstance(adapter, tuple) and adapter and adapter[0] == "config":
                    continue  # per-layer attention config (ComfyUI #16880) — не вес
                if isinstance(adapter, tuple):
                    if adapter and adapter[0] not in ("diff",) and len(adapter) == 2:
                        rejected.append((name, str(adapter[0])))
                        continue
                elif kind not in ADDITIVE:
                    rejected.append((name, kind))
                    continue
                elif _has_dora(adapter):
                    rejected.append((name, "DoRA"))
                    continue
                self.patches.setdefault(name, []).append((strength, adapter, 1.0, offset, function))
                used += 1
            text = sum(1 for k in lora if k.startswith(("lora_te", "text_encoder", "te_", "lora_te1", "lora_te2")))
            if rejected:
                raise ValueError(f"{Path(path).name}: адаптеры {sorted({r[1] for r in rejected})} зависят от исходного "
                                 f"веса и не сливаются по строкам shard (пример: {rejected[0][0]}). Поддерживаются "
                                 "LoRA, LoHa, LoKr и diff")
            if not used:
                raise ValueError(f"{Path(path).name}: ни один ключ LoRA не подошёл модели — LoRA от другой модели?")
            self.report.append(dict(path=path, strength=strength, parameters=used, text_encoder_keys_ignored=text))
            del lora
        self.applied = 0

    def __bool__(self):
        return bool(self.patches)

    def has(self, name):
        return (self.prefix + name) in self.patches

    def full_delta(self, name, shape, device):
        """Полная FP32 дельта параметра (name — без префикса) или None."""
        import comfy.lora
        patches = self.patches.get(self.prefix + name)
        if not patches:
            return None
        zero = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
        out = comfy.lora.calculate_weight(patches, zero, self.prefix + name, intermediate_dtype=torch.float32)
        if not torch.isfinite(out).all():
            raise FloatingPointError(f"LoRA: не конечная дельта {name}")
        if not out.abs().amax() > 0:
            # адаптеры ComfyUI при ошибке формы только пишут в лог и оставляют вес как был
            raise ValueError(f"LoRA для {name} не применилась (форма адаптера не совпадает с весом {list(shape)})")
        self.applied += 1
        return out

    def delta(self, name, a, b, shape, device=None):
        """Совместимо с wan_lora.LoraMerger: строки [a:b] FP32 дельты."""
        full = self.full_delta(name, shape, device or torch.device("cpu"))
        return None if full is None else full[a:b]


def _has_dora(adapter):
    """dora_scale: LoRA — weights[4], LoHa / LoKr — последний элемент (comfy.weight_adapter)."""
    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (list, tuple)) or not weights:
        return False
    scale = weights[4] if type(adapter).__name__ == "LoRAAdapter" and len(weights) > 4 else weights[-1]
    return isinstance(scale, torch.Tensor)
