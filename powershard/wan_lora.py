"""LoRA для FSDP Wan: слияние в ЛОКАЛЬНЫЕ строки shard при загрузке.

Ни один rank не строит полный вес: для Linear W[a:b] += s * up[a:b] @ down.
Форматы (как в ComfyUI для Wan): ``diffusion_model.<module>`` и ``lora_unet_<module_with_underscores>``
с суффиксами lora_up/lora_down(+alpha), lora_A/lora_B, lora.up/lora.down, а также
``.diff`` (полная дельта веса, напр. norm/modulation у lightx2v) и ``.diff_b`` (дельта bias).
LoHa/LoKr/DoRA/LoCon-mid не поддерживаются и вызывают явную ошибку, а не тихий пропуск.
Планирование ключей не требует torch.
"""
import json
import math
from pathlib import Path
import struct

SUFFIXES = (
    (".lora_up.weight", "up"), (".lora_down.weight", "down"),
    (".lora_B.weight", "up"), (".lora_A.weight", "down"),
    (".lora.up.weight", "up"), (".lora.down.weight", "down"),
    (".alpha", "alpha"), (".diff_b", "diff_b"), (".diff", "diff"),
    (".lora_mid.weight", "unsupported:LoCon mid"), (".dora_scale", "unsupported:DoRA"),
    (".hada_w1_a", "unsupported:LoHa"), (".hada_w1_b", "unsupported:LoHa"),
    (".hada_w2_a", "unsupported:LoHa"), (".hada_w2_b", "unsupported:LoHa"),
    (".lokr_w1", "unsupported:LoKr"), (".lokr_w2", "unsupported:LoKr"),
    (".lokr_w1_a", "unsupported:LoKr"), (".lokr_w1_b", "unsupported:LoKr"),
    (".lokr_w2_a", "unsupported:LoKr"), (".lokr_w2_b", "unsupported:LoKr"),
)
PREFIXES = ("model.diffusion_model.", "diffusion_model.", "lora_unet__", "lora_unet_", "lycoris_")
TEXT_ENCODER_PREFIXES = ("lora_te", "text_encoders.", "te_", "text_encoder.", "lora_te1_", "lora_te2_")


def safetensors_header(path):
    with Path(path).open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"Обрезанный safetensors: {path}")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= 64 * 1024 * 1024:
            raise ValueError(f"Недопустимый заголовок safetensors: {path}")
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


def split_suffix(key):
    for suffix, role in SUFFIXES:
        if key.endswith(suffix):
            return key[:-len(suffix)], role
    return None, None


def plan_lora(lora_header, model_tensors, label="LoRA"):
    """Сопоставить ключи LoRA параметрам модели. model_tensors: name -> {"shape": [...]}.

    Возвращает dict(entries={param_name: {role: lora_key}}, ignored=[...], rank=...).
    """
    shapes = {k: list(v["shape"]) for k, v in model_tensors.items()}
    underscored = {}
    for name in shapes:
        module = name[:-len(".weight")] if name.endswith(".weight") else name
        underscored[module.replace(".", "_")] = module
    groups, ignored, unknown = {}, [], []
    for key in lora_header:
        if key.startswith(TEXT_ENCODER_PREFIXES):
            ignored.append(key)
            continue
        base, role = split_suffix(key)
        if base is None:
            unknown.append(key)
            continue
        if role.startswith("unsupported:"):
            raise ValueError(f"{label}: формат {role.split(':', 1)[1]} ({key}) не поддерживается PowerShard Wan; "
                             "используйте обычную LoRA (lora_up/lora_down или lora_A/lora_B)")
        module = None
        for prefix in PREFIXES:
            if base.startswith(prefix):
                rest = base[len(prefix):]
                module = underscored.get(rest) if prefix.startswith(("lora_unet", "lycoris")) else rest
                if module is not None and (module + ".weight") not in shapes and module not in shapes:
                    module = None
                break
        else:
            module = base if (base + ".weight") in shapes or base in shapes else None
        if module is None:
            unknown.append(key)
            continue
        groups.setdefault(module, {})[role] = key
    if unknown:
        raise ValueError(f"{label}: {len(unknown)} ключей не соответствуют модели Wan (пример: {unknown[:6]}). "
                         "Возможно, LoRA от другой модели (Wan 2.1 vs 2.2 5B, другой dim) или в формате diffusers.")
    entries, ranks = {}, set()
    for module, roles in groups.items():
        if ("up" in roles) != ("down" in roles):
            raise ValueError(f"{label}: у {module} нет пары up/down")
        if "alpha" in roles and "up" not in roles:
            raise ValueError(f"{label}: alpha без up/down у {module}")
        if "up" in roles or "diff" in roles:
            target = module + ".weight" if module + ".weight" in shapes else module if module in shapes else None
            if target is None:
                raise ValueError(f"{label}: нет параметра для {module}")
            shape = shapes[target]
            if "up" in roles:
                up, down = lora_header[roles["up"]]["shape"], lora_header[roles["down"]]["shape"]
                rank = up[1] if len(up) > 1 else 1
                if up[0] != shape[0] or down[0] != rank or math.prod(down[1:]) != math.prod(shape[1:]):
                    raise ValueError(f"{label}: форма LoRA {module} up{up}/down{down} не совпадает с весом {shape}")
                ranks.add(rank)
                entries.setdefault(target, {}).update(up=roles["up"], down=roles["down"], alpha=roles.get("alpha"))
            if "diff" in roles:
                if lora_header[roles["diff"]]["shape"] != shape:
                    raise ValueError(f"{label}: diff {module} имеет форму {lora_header[roles['diff']]['shape']}, вес {shape}")
                entries.setdefault(target, {})["diff"] = roles["diff"]
        if "diff_b" in roles:
            target = module + ".bias"
            if target not in shapes or lora_header[roles["diff_b"]]["shape"] != shapes[target]:
                raise ValueError(f"{label}: diff_b {module} не совпадает с bias")
            entries.setdefault(target, {})["diff"] = roles["diff_b"]
    if not entries:
        raise ValueError(f"{label}: не найдено ни одного применимого ключа для Wan")
    return dict(entries=entries, ignored=ignored, ranks=sorted(ranks))


class LoraMerger:
    """Worker-side: дельты для локальных строк [a:b], FP32, без полного веса."""

    def __init__(self, specs, model_tensors, label_prefix=""):
        from safetensors import safe_open
        self.sources, self.report = [], []
        for spec in specs:
            path = spec["path"]
            plan = plan_lora(safetensors_header(path), model_tensors, label=label_prefix + Path(path).name)
            handle = safe_open(path, framework="pt", device="cpu")
            self.sources.append((handle, plan["entries"], float(spec["strength"])))
            self.report.append(dict(path=path, strength=float(spec["strength"]), apply_to=spec.get("apply_to"),
                                    parameters=len(plan["entries"]), ignored_text_encoder_keys=len(plan["ignored"]),
                                    ranks=plan["ranks"]))
        self.applied = 0

    def __bool__(self):
        return bool(self.sources)

    def delta(self, name, a, b, shape, device=None):
        """FP32 дельта строк [a:b]; device — где считать up@down (GPU rank при загрузке)."""
        import torch
        device = device or torch.device("cpu")
        total = None
        for handle, entries, strength in self.sources:
            entry = entries.get(name)
            if not entry:
                continue
            rows = b - a
            value = torch.zeros((rows,) + tuple(shape[1:]), dtype=torch.float32, device=device)
            if "diff" in entry:
                value += handle.get_slice(entry["diff"])[a:b].to(device).float().reshape(value.shape)
            if "up" in entry:
                up = handle.get_slice(entry["up"])[a:b].to(device).float().reshape(rows, -1)
                down = handle.get_tensor(entry["down"]).to(device).float()
                rank = down.shape[0]
                scale = 1.0
                if entry.get("alpha"):
                    scale = float(handle.get_tensor(entry["alpha"]).float().reshape(-1)[0]) / rank
                if rows:
                    value += (up @ down.reshape(rank, -1)).reshape(value.shape) * scale
            total = value * strength if total is None else total + value * strength
            self.applied += 1
        return total
