"""Чтение заголовков без torch; строгая проверка формата перед worker startup."""
import hashlib
import json
import math
from pathlib import Path
import struct

BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "F16": 2, "BF16": 2,
         "I32": 4, "F32": 4, "I64": 8, "F64": 8}


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        with self.path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("Обрезанный safetensors")
            self.header_size = struct.unpack("<Q", raw)[0]
            if not 2 <= self.header_size <= 32 * 1024 * 1024:
                raise ValueError("Недопустимый размер заголовка safetensors")
            raw = f.read(self.header_size)
        self.header_hash = hashlib.sha256(raw).hexdigest()
        self.header = json.loads(raw)
        self.metadata = self.header.get("__metadata__", {})
        self.tensors = {k: v for k, v in self.header.items() if k != "__metadata__"}
        self.data_start = self.header_size + 8
        file_size = self.path.stat().st_size
        intervals = []
        for name, desc in self.tensors.items():
            shape, dt = desc["shape"], desc["dtype"]
            a, b = desc["data_offsets"]
            if dt not in BYTES or any(type(n) != int or n < 0 for n in shape):
                raise ValueError(f"Неподдерживаемый dtype/shape: {name}")
            if a < 0 or b - a != math.prod(shape) * BYTES[dt] or self.data_start + b > file_size:
                raise ValueError(f"Некорректный диапазон tensor: {name}")
            intervals.append((a, b))
        end = 0
        for a, b in sorted(intervals):
            if a != end:
                raise ValueError("Перекрывающиеся tensors или разрыв safetensors")
            end = b
        if self.data_start + end != file_size:
            raise ValueError("Размер файла не соответствует заголовку")

    def read_json_tensor(self, name):
        desc = self.tensors[name]
        a, b = desc["data_offsets"]
        if desc["dtype"] != "U8" or b - a > 65536:
            raise ValueError(f"Неверные quant metadata: {name}")
        with self.path.open("rb") as f:
            f.seek(self.data_start + a)
            return json.loads(f.read(b - a))

    def model_config(self):
        return infer_h3_config(self.tensors)

    def identity(self):
        st = self.path.stat()
        return {"path": str(self.path), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                "header_sha256": self.header_hash}

    def quantization(self):
        configs = {}
        for name, d in self.tensors.items():
            if d["dtype"] == "I8":
                if not name.endswith(".weight") or len(d["shape"]) != 2:
                    raise ValueError(f"INT8 вне Linear: {name}")
                module = name[:-7]
                conf = self.read_json_tensor(module + ".comfy_quant")
                if conf.get("format") != "int8_tensorwise":
                    raise ValueError(f"Неподдерживаемое квантование {module}: {conf}")
                params = conf.get("params", {})
                gs = int(conf.get("convrot_groupsize", params.get("convrot_groupsize", 256)))
                convrot = bool(conf.get("convrot", params.get("convrot", False)))
                if convrot and (gs < 4 or gs & (gs - 1) or (gs.bit_length() - 1) % 2 or d["shape"][1] % gs):
                    raise ValueError(f"Некорректная ConvRot-группа: {module}")
                scale = self.tensors[module + ".weight_scale"]
                if scale["shape"] != [d["shape"][0], 1] or scale["dtype"] != "F32":
                    raise ValueError(f"Ожидался row-wise FP32 scale: {module}")
                configs[module] = {"convrot": convrot, "group_size": gs}
        return configs


def infer_h3_config(tensors):
    def shape(k):
        return tensors[k]["shape"]
    def blocks(prefix):
        ids = {int(k[len(prefix):].split(".")[0]) for k in tensors if k.startswith(prefix)}
        if ids != set(range(len(ids))):
            raise ValueError(f"Непоследовательные блоки {prefix}")
        return len(ids)
    try:
        head_dim = shape("blocks.0.attn.q_norm.weight")[0]
        conf = dict(hidden_size=shape("video_patch_proj.weight")[0], num_layers=blocks("blocks."),
                    token_refiner_num_layers=blocks("token_refiner.blocks."),
                    num_attention_heads=shape("blocks.0.attn.qkv_proj.weight")[0] // (3 * head_dim),
                    attention_head_dim=head_dim, ffn_hidden_size=shape("blocks.0.mlp.fc1.weight")[0] // 2,
                    text_dim=shape("condition_proj.weight")[1],
                    latents_dim=shape("final_layer.video_out.weight")[0] // 4,
                    audio_latents_dim=shape("final_layer.audio_out.weight")[0],
                    rope_inv_freq_len=shape("rope.inv_freq")[0])
        if "adaln_t_table" not in tensors:
            raise ValueError("Поддерживаются Pruned checkpoints с adaln_t_table")
        conf.update(adaln_curve_grid=shape("adaln_t_table")[0], time_embed_dim=shape("adaln_t_table")[1])
        if any("to_gate_compress" in k for k in tensors):
            raise ValueError("VSA/PDD checkpoints пока не поддерживаются")
        return conf
    except KeyError as e:
        raise ValueError(f"Это не поддерживаемый native H3 checkpoint; отсутствует {e}") from e


def memory_plan(tensors, quantized=False, world_size=1):
    groups, stored, target = {}, 0, 0
    for k, v in tensors.items():
        n = math.prod(v["shape"])
        stored += n * BYTES[v["dtype"]]
        if k.endswith(".comfy_quant"):
            continue
        # FP32 islands остаются FP32; остальные float -> FP16.
        b = n * (1 if v["dtype"] == "I8" else 4 if v["dtype"] == "F32" or ".adaln_proj." in k else 2)
        target += b
        parts = k.split(".")
        group = ".".join(parts[:2]) if parts[0] == "blocks" else "root_and_aux"
        groups[group] = groups.get(group, 0) + b
    max_group = max(groups.values(), default=0)
    return {"checkpoint_tensor_bytes": stored, "converted_storage_bytes": target,
            "world_size": world_size,
            "shard_bytes_lower_bound": math.ceil(target / world_size), "largest_group_bytes_upper_bound": max_group,
            "note_ru": "К shard добавляются padding, buffers, активный all-gather, allocator/NCCL, activations, attention и dequant. Не является гарантией размещения."}
