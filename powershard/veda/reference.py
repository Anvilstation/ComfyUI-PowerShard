# Переработано из Veda-on-ComfyUI (MIT, (c) 2026 veda-sparse), veda_comfy/core/reference.py. См. NOTICE.md.
"""Точная блочно-разреженная attention в FP32 — эталон для self-test бэкендов (O(N^2), только малые задачи)."""
import torch

from . import tiling


def block_sparse_attention(q, k, v, block_mask, layout, softmax_scale=None):
    """q, k, v: [N, H', D] в порядке тайлов; block_mask: [H', n_tiles, n_tiles]. Строки паддинга = 0."""
    tile = tiling.TILE_SIZE
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    allowed = block_mask.repeat_interleave(tile, 1).repeat_interleave(tile, 2)
    allowed = allowed & layout.slot_valid.bool()[None, None, :]
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    out = torch.einsum("hqk,khd->qhd", probs, v.float())
    return out * layout.slot_valid.to(out.dtype)[:, None, None]
