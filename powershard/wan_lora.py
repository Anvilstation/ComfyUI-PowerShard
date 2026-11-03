"""LoRA для FSDP Wan / LTX: слияние в ЛОКАЛЬНЫЕ строки shard при загрузке.

Ни один rank не строит полный вес Linear: для LoRA W[a:b] += s * up[a:b] @ down.
Префиксы: ``diffusion_model.`` / ``model.diffusion_model.`` / ``lora_unet_`` / ``lycoris_`` / без префикса.
Форматы (как comfy.weight_adapter): lora_up/lora_down(+alpha), lora_A/lora_B (kohya/diffusers/PEFT
``.lora_A.default.weight``/mochi без ``.weight``), ``_lora.up/down``, ``lora.up/down``,
``lora_linear_layer.up/down``; LoHa (hada_w*), LoKr (lokr_w1/w2 и их a/b); ``.diff``/``.diff_b`` (полная
дельта веса/bias, напр. lightx2v). DoRA, LoCon-mid и Tucker (t1/t2) требуют полного веса — явная ошибка.
Ключи, не подходящие модели, пропускаются с предупреждением (как «lora key not loaded» в ComfyUI);
ошибка — только если не подошёл ни один. Планирование ключей не требует torch.
"""
import json
import math
from pathlib import Path
import struct
import warnings

SUFFIXES = (
    (".lora_A.default.weight", "down"), (".lora_B.default.weight", "up"),          # PEFT (qwen/peft default)
    (".lora_linear_layer.down.weight", "down"), (".lora_linear_layer.up.weight", "up"),
    ("_lora.down.weight", "down"), ("_lora.up.weight", "up"),                      # старый diffusers
    (".lora_up.weight", "up"), (".lora_down.weight", "down"),
    (".lora_B.weight", "up"), (".lora_A.weight", "down"),
    (".lora.up.weight", "up"), (".lora.down.weight", "down"),
    (".lora_B", "up"), (".lora_A", "down"),                                        # mochi
    (".alpha", "alpha"), (".diff_b", "diff_b"), (".diff", "diff"),
    (".hada_w1_a", "hada_w1_a"), (".hada_w1_b", "hada_w1_b"),
    (".hada_w2_a", "hada_w2_a"), (".hada_w2_b", "hada_w2_b"),
    (".lokr_w1_a", "lokr_w1_a"), (".lokr_w1_b", "lokr_w1_b"),
    (".lokr_w2_a", "lokr_w2_a"), (".lokr_w2_b", "lokr_w2_b"),
    (".lokr_w1", "lokr_w1"), (".lokr_w2", "lokr_w2"),
    (".lora_mid.weight", "unsupported:LoCon mid (Tucker)"), (".dora_scale", "unsupported:DoRA"),
    (".hada_t1", "unsupported:LoHa Tucker"), (".hada_t2", "unsupported:LoHa Tucker"),
    (".lokr_t2", "unsupported:LoKr Tucker"),
    (".reshape_weight", "unsupported:reshape_weight"), (".set_weight", "unsupported:set_weight"),
)
PREFIXES = ("model.diffusion_model.", "diffusion_model.", "lora_unet__", "lora_unet_", "lycoris_")
TEXT_ENCODER_PREFIXES = ("lora_te", "text_encoders.", "te_", "text_encoder.", "lora_te1_", "lora_te2_")
LOKR_ROLES = ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b")
LOHA_ROLES = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")


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


def _module_for(base, shapes, underscored):
    for prefix in PREFIXES:
        if base.startswith(prefix):
            rest = base[len(prefix):]
            module = underscored.get(rest) if prefix.startswith(("lora_unet", "lycoris")) else rest
            if module is not None and (module + ".weight") not in shapes and module not in shapes:
                return None
            return module
    return base if (base + ".weight") in shapes or base in shapes else None


def _pair(header, roles, single, a, b):
    """Форма [out, in] фактора LoKr: прямой w или произведение a @ b."""
    if single in roles:
        shape = header[roles[single]]["shape"]
        return (shape[0], math.prod(shape[1:]) if len(shape) > 1 else 1), None
    if a in roles and b in roles:
        sa, sb = header[roles[a]]["shape"], header[roles[b]]["shape"]
        return (sa[0], math.prod(sb[1:]) if len(sb) > 1 else 1), sb[0]
    return None, None


def plan_lora(lora_header, model_tensors, label="LoRA"):
    """Сопоставить ключи LoRA параметрам модели. model_tensors: name -> {"shape": [...]}.

    Возвращает dict(entries={param_name: {role: lora_key}}, ignored=[...], skipped=[...], rank=...).
    """
    shapes = {k: list(v["shape"]) for k, v in model_tensors.items()}
    underscored = {}
    for name in shapes:
        module = name[:-len(".weight")] if name.endswith(".weight") else name
        underscored[module.replace(".", "_")] = module
    groups, ignored, skipped = {}, [], []
    for key in lora_header:
        if key.startswith(TEXT_ENCODER_PREFIXES):
            ignored.append(key)
            continue
        base, role = split_suffix(key)
        if base is None:
            skipped.append((key, "неизвестный суффикс"))
            continue
        if role.startswith("unsupported:"):
            raise ValueError(f"{label}: формат {role.split(':', 1)[1]} ({key}) требует полного веса и не сливается по "
                             "строкам shard; используйте обычную LoRA/LoHa/LoKr")
        module = _module_for(base, shapes, underscored)
        if module is None:
            skipped.append((key, "нет такого модуля в модели"))
            continue
        groups.setdefault(module, {})[role] = key
    entries, ranks = {}, set()

    def skip(module, roles, why):
        skipped.extend((k, why) for k in roles.values())

    for module, roles in groups.items():
        target = module + ".weight" if module + ".weight" in shapes else module if module in shapes else None
        if "diff_b" in roles:
            bias = module + ".bias"
            if bias not in shapes or lora_header[roles["diff_b"]]["shape"] != shapes[bias]:
                skip(module, {"diff_b": roles.pop("diff_b")}, "diff_b не совпадает с bias")
            else:
                entries.setdefault(bias, {})["diff"] = roles.pop("diff_b")
        if target is None:
            if set(roles) - {"alpha"}:
                skip(module, roles, "нет параметра weight")
            continue
        shape = shapes[target]
        if "diff" in roles:
            if lora_header[roles["diff"]]["shape"] != shape and \
                    math.prod(lora_header[roles["diff"]]["shape"]) != math.prod(shape):
                skip(module, {"diff": roles["diff"]}, f"diff другой формы ({shape})")
            else:
                entries.setdefault(target, {})["diff"] = roles["diff"]
        if "up" in roles or "down" in roles:
            if ("up" in roles) != ("down" in roles):
                skip(module, {k: roles[k] for k in ("up", "down") if k in roles}, "нет пары up/down")
            else:
                up, down = lora_header[roles["up"]]["shape"], lora_header[roles["down"]]["shape"]
                rank = up[1] if len(up) > 1 else 1
                if up[0] != shape[0] or down[0] != rank or math.prod(down[1:]) != math.prod(shape[1:]):
                    skip(module, {"up": roles["up"], "down": roles["down"]},
                         f"форма up{up}/down{down} не совпадает с весом {shape}")
                else:
                    ranks.add(rank)
                    entries.setdefault(target, {}).update(up=roles["up"], down=roles["down"], alpha=roles.get("alpha"))
        if any(r in roles for r in LOHA_ROLES):
            if not all(r in roles for r in LOHA_ROLES):
                skip(module, {r: roles[r] for r in LOHA_ROLES if r in roles}, "неполная LoHa")
            else:
                w1a, w1b = lora_header[roles["hada_w1_a"]]["shape"], lora_header[roles["hada_w1_b"]]["shape"]
                if w1a[0] != shape[0] or math.prod(w1b[1:]) != math.prod(shape[1:]):
                    skip(module, {r: roles[r] for r in LOHA_ROLES}, f"форма LoHa не совпадает с {shape}")
                else:
                    entries.setdefault(target, {})["loha"] = dict({r: roles[r] for r in LOHA_ROLES},
                                                                   alpha=roles.get("alpha"))
        if any(r in roles for r in LOKR_ROLES):
            (o1, i1), dim1 = _pair(lora_header, roles, "lokr_w1", "lokr_w1_a", "lokr_w1_b") if \
                _pair(lora_header, roles, "lokr_w1", "lokr_w1_a", "lokr_w1_b")[0] else ((None, None), None)
            (o2, i2), dim2 = _pair(lora_header, roles, "lokr_w2", "lokr_w2_a", "lokr_w2_b") if \
                _pair(lora_header, roles, "lokr_w2", "lokr_w2_a", "lokr_w2_b")[0] else ((None, None), None)
            parts = {r: roles[r] for r in LOKR_ROLES if r in roles}
            if o1 is None or o2 is None or o1 * o2 != shape[0] or i1 * i2 != math.prod(shape[1:]):
                skip(module, parts, f"форма LoKr не совпадает с {shape}")
            else:
                entries.setdefault(target, {})["lokr"] = dict(parts, alpha=roles.get("alpha"),
                                                              dim=dim2 if dim2 is not None else dim1)
    if not entries:
        raise ValueError(f"{label}: ни один ключ LoRA не соответствует модели (пример: {skipped[:6]}). "
                         "Возможно, LoRA от другой модели (Wan 2.1 vs 2.2 5B, LTX, другой dim)")
    if skipped:
        warnings.warn(f"{label}: пропущено {len(skipped)} ключей LoRA, не подходящих модели "
                      f"(пример: {skipped[:4]}). Применено параметров: {len(entries)}")
    return dict(entries=entries, ignored=ignored, skipped=[k for k, _ in skipped], ranks=sorted(ranks))


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
                                    skipped_keys=len(plan["skipped"]),
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
            if "loha" in entry and rows:
                spec = entry["loha"]
                w1a = handle.get_slice(spec["hada_w1_a"])[a:b].to(device).float()
                w2a = handle.get_slice(spec["hada_w2_a"])[a:b].to(device).float()
                w1b = handle.get_tensor(spec["hada_w1_b"]).to(device).float()
                w2b = handle.get_tensor(spec["hada_w2_b"]).to(device).float()
                scale = 1.0
                if spec.get("alpha"):
                    scale = float(handle.get_tensor(spec["alpha"]).float().reshape(-1)[0]) / w1b.shape[0]
                m1 = w1a @ w1b.reshape(w1b.shape[0], -1)
                m2 = w2a @ w2b.reshape(w2b.shape[0], -1)
                value += (m1 * m2).reshape(value.shape) * scale
            if "lokr" in entry and rows:
                spec = entry["lokr"]

                def factor(single, fa, fb):
                    if spec.get(single):
                        return handle.get_tensor(spec[single]).to(device).float()
                    fa_t = handle.get_tensor(spec[fa]).to(device).float()
                    fb_t = handle.get_tensor(spec[fb]).to(device).float()
                    return fa_t @ fb_t.reshape(fb_t.shape[0], -1)
                w1 = factor("lokr_w1", "lokr_w1_a", "lokr_w1_b")
                w2 = factor("lokr_w2", "lokr_w2_a", "lokr_w2_b")
                w1, w2 = w1.reshape(w1.shape[0], -1), w2.reshape(w2.shape[0], -1)
                scale = 1.0
                if spec.get("alpha") and spec.get("dim"):
                    scale = float(handle.get_tensor(spec["alpha"]).float().reshape(-1)[0]) / float(spec["dim"])
                # Строки kron(w1, w2): r = r1 * o2 + r2 -> только нужные строки, без полного веса.
                o2 = w2.shape[0]
                index = torch.arange(a, b, device=device)
                part = w1[index // o2][:, :, None] * w2[index % o2][:, None, :]
                value += part.reshape(value.shape) * scale
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
