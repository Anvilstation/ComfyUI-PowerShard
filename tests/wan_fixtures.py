"""Tiny Wan headers/files без torch: safetensors пишется вручную (JSON header + raw bytes)."""
import json
import math
import struct
from pathlib import Path

BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1}


def tiny_wan_shapes(dim=256, ffn=512, layers=2, in_dim=16, out_dim=16, text_dim=32, freq_dim=16, i2v=False, ref=False):
    shapes = {
        "patch_embedding.weight": [dim, in_dim, 1, 2, 2], "patch_embedding.bias": [dim],
        "text_embedding.0.weight": [dim, text_dim], "text_embedding.0.bias": [dim],
        "text_embedding.2.weight": [dim, dim], "text_embedding.2.bias": [dim],
        "time_embedding.0.weight": [dim, freq_dim], "time_embedding.0.bias": [dim],
        "time_embedding.2.weight": [dim, dim], "time_embedding.2.bias": [dim],
        "time_projection.1.weight": [6 * dim, dim], "time_projection.1.bias": [6 * dim],
        "head.head.weight": [out_dim * 4, dim], "head.head.bias": [out_dim * 4], "head.modulation": [1, 2, dim],
    }
    for i in range(layers):
        p = f"blocks.{i}."
        for kind in ("self_attn", "cross_attn"):
            for proj in ("q", "k", "v", "o"):
                shapes[p + f"{kind}.{proj}.weight"] = [dim, dim]
                shapes[p + f"{kind}.{proj}.bias"] = [dim]
            shapes[p + f"{kind}.norm_q.weight"] = [dim]
            shapes[p + f"{kind}.norm_k.weight"] = [dim]
        if i2v:
            for proj in ("k_img", "v_img"):
                shapes[p + f"cross_attn.{proj}.weight"] = [dim, dim]
                shapes[p + f"cross_attn.{proj}.bias"] = [dim]
            shapes[p + "cross_attn.norm_k_img.weight"] = [dim]
        shapes[p + "norm3.weight"] = [dim]
        shapes[p + "norm3.bias"] = [dim]
        shapes[p + "ffn.0.weight"] = [ffn, dim]
        shapes[p + "ffn.0.bias"] = [ffn]
        shapes[p + "ffn.2.weight"] = [dim, ffn]
        shapes[p + "ffn.2.bias"] = [dim]
        shapes[p + "modulation"] = [1, 6, dim]
    if i2v:
        shapes.update({"img_emb.proj.0.weight": [1280], "img_emb.proj.0.bias": [1280],
                       "img_emb.proj.1.weight": [1280, 1280], "img_emb.proj.1.bias": [1280],
                       "img_emb.proj.3.weight": [dim, 1280], "img_emb.proj.3.bias": [dim],
                       "img_emb.proj.4.weight": [dim], "img_emb.proj.4.bias": [dim]})
    if ref:
        shapes.update({"ref_conv.weight": [dim, in_dim, 2, 2], "ref_conv.bias": [dim]})
    return shapes


def write_safetensors(path, tensors, metadata=None):
    """tensors: name -> (dtype, shape, raw bytes or None for zeros)."""
    header, chunks, offset = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        size = math.prod(shape) * BYTES[dtype]
        data = bytes(size) if data is None else data
        assert len(data) == size, name
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        chunks.append(data)
        offset += size
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))
    return path


def write_tiny_wan(path, prefix="", dtype="F16", fp8_scaled=False, extra=None, **kw):
    tensors = {}
    for name, shape in tiny_wan_shapes(**kw).items():
        if fp8_scaled and name.startswith("blocks.") and name.endswith(".weight") and len(shape) == 2:
            tensors[prefix + name] = ("F8_E4M3", shape, None)
            tensors[prefix + name[:-len(".weight")] + ".scale_weight"] = ("F32", [], None)
        else:
            tensors[prefix + name] = (dtype, shape, None)
    if fp8_scaled:
        tensors[prefix + "scaled_fp8"] = ("F8_E4M3", [0], None)
    tensors.update(extra or {})
    return write_safetensors(path, tensors)
