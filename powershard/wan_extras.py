"""Wan варианты со своей структурой внимания: Animate2 (pose branch), InfiniteTalk/MultiTalk
(аудио cross-attention + карта говорящих), WanDancer music encoder.

Всё работает на локальных строках sequence shard и соблюдает правило FSDP: каждый unit
вызывается одинаковое число раз в одинаковом порядке на всех rank. Pose branch Animate2
считается внутри того же вызова блока, что и генерация (веса собираются один раз).
"""
from collections import OrderedDict
import types
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .config import shard_bounds


def _wm():
    from . import wan_model
    return wan_model


# ------------------------------------------------------------------ WanDancer
def music_attention_forward(self, x, freqs):
    """MusicSelfAttention (2 слоя x 256 dim, ~F*1 токенов): FP32 SDPA с тем же RoPE."""
    rope = _wm().rope_apply
    b, s = x.shape[:2]
    n, d = self.num_heads, self.head_dim
    q = rope(self.q_proj(x).float().view(b, s, n, d), freqs)
    k = rope(self.k_proj(x).float().view(b, s, n, d), freqs)
    v = self.v_proj(x).float().view(b, s, n, d)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return self.out_proj(out.transpose(1, 2).reshape(b, s, n * d)).float()


def music_embedding(net, audio_embed, latent_frames):
    """[B, frames, 35] -> [B, F*8, dim]: projection, encoder (FSDP units), bilinear interpolate как native."""
    music = net.music_projection(audio_embed.float()).float()
    ids = torch.arange(music.shape[1], device=music.device, dtype=torch.float32).reshape(1, -1, 1)
    freqs = net.music_rope_embedder(ids).movedim(1, 2).float()
    for layer in net.music_encoder:
        music = layer(music, freqs).float()
    return F.interpolate(music.unsqueeze(1), size=(latent_frames * 8, net.dim), mode="bilinear").squeeze(1)


# ----------------------------------------------------------------- Animate2
def joint_v_scale(tensors, collective):
    from .fp16_safe import power2_scale
    values = [t.float().abs().amax() for t in tensors if t.numel()]
    maximum = torch.stack(values).amax() if values else tensors[0].new_zeros((), dtype=torch.float32)
    if collective:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return power2_scale(maximum / 16384.)


def pose_tail_attention(attn, q, k, v, kp, vp, segments, hw, sq, sk, sv, restore):
    """Кадр g видит все токены генерации + кадр g-1 pose branch (кадр 0 — слот референса, без хвоста).

    q — строки segments; k/v — вся генерация; kp/vp — вся pose последовательность.
    Буфер [gen | hw] выделяется один раз, переписывается только хвост (как native).
    """
    run = _wm().run_attention
    length = k.shape[1]
    kbuf = k.new_empty((k.shape[0], length + hw) + tuple(k.shape[2:]))
    vbuf = v.new_empty((v.shape[0], length + hw) + tuple(v.shape[2:]))
    kbuf[:, :length], vbuf[:, :length] = k, v
    out = q.new_zeros(q.shape[:-1] + (v.shape[-1],), dtype=torch.float32 if restore else torch.float16)
    for start, stop, g in segments:
        if g == 0:
            kk, vv = k, v
        else:
            kbuf[:, length:] = kp[:, (g - 1) * hw:g * hw]
            vbuf[:, length:] = vp[:, (g - 1) * hw:g * hw]
            kk, vv = kbuf, vbuf
        out[:, start:stop] = run(attn, q[:, start:stop], kk, vv, sq, sk, attn._ps_group, sv=sv, restore_v=restore)
    return out


def animate2_attention(attn, q, k, v, kp, vp, plan):
    """Распределённая forward_gen attention: token (all-gather K/V обеих ветвей) или Ulysses (all-to-all)."""
    from .telemetry import region
    wm = _wm()
    state = attn._ps_sequence
    heads = attn.num_heads
    sq, sk = attn._ps_qk_scales[:2]
    hw, total, total_p = plan["hw"], plan["total"], plan["pose"]["total"]
    if not state["enabled"]:
        return pose_tail_attention(attn, q, k, v, kp, vp, wm.frame_segments(0, total, hw), hw, sq, sk, None, True)
    fp16_wire = state["comm_dtype"] == "fp16"
    sv = None
    if fp16_wire:
        q, k, kp = wm.scaled_half(q, sq), wm.scaled_half(k, sk), wm.scaled_half(kp, sk)
        sv = joint_v_scale((v, vp), True)
        state["scale_collectives"] += 1
        v, vp = wm.scaled_half(v, sv), wm.scaled_half(vp, sv)
    else:
        q, k, v, kp, vp = q.float(), k.float(), v.float(), kp.float(), vp.float()
    if state["ulysses"]:
        with region("wan_ulysses_QKV_all_to_all"):
            (q, k, v), padded, exchanged = wm.ulysses_to_sequence((q, k, v), total, heads)
            (kp, vp), _, exchanged_pose = wm.ulysses_to_sequence((kp, vp), total_p, heads)
        state["padded_heads"] = padded
        state["kv_collectives"] += 2
        state["kv_gathered_bytes"] += exchanged + exchanged_pose
        out = pose_tail_attention(attn, q, k, v, kp, vp, wm.frame_segments(0, total, hw), hw, sq, sk, sv,
                                  not fp16_wire)
        with region("wan_ulysses_output_all_to_all"):
            out, exchanged = wm.ulysses_to_heads(out, total, heads)
        state["out_collectives"] += 1
        state["out_exchanged_bytes"] += exchanged
        if fp16_wire:
            out = out.float() * sv
    else:
        with region("wan_sequence_KV_all_gather"):
            kv = wm.gather_tokens(torch.stack((k, v), dim=2), total)
            kvp = wm.gather_tokens(torch.stack((kp, vp), dim=2), total_p)
        state["kv_collectives"] += 2
        state["kv_gathered_bytes"] += kv.numel() * kv.element_size() + kvp.numel() * kvp.element_size()
        (k, v), (kp, vp) = kv.unbind(2), kvp.unbind(2)
        a, b = plan["rows"]
        out = pose_tail_attention(attn, q, k, v, kp, vp, wm.frame_segments(a, b, hw), hw, sq, sk, sv, True)
    state["effective_comm_dtype"] = "float16" if fp16_wire else "float32"
    return out


def animate2_block_forward(self, x, e0, freqs, context, context_img_len, tokens, plan=None):
    """WanAnimate2Block: forward_pose (или kv_from_input из cache) + forward_gen одним вызовом FSDP unit.

    Возвращает (x, x_pose_next | None).
    """
    wm = _wm()
    attn = self.self_attn
    batch, n, _ = x.shape
    heads, dim = attn.num_heads, attn.head_dim
    pose = plan.get("pose") if plan else None
    kp = vp = new_pose = None
    if pose is not None:
        ep = (self.modulation.float().unsqueeze(0) + pose["e0"].float()).unbind(2)
        xp = pose["x"].float()
        hidden = torch.addcmul(ep[0], self.norm1(xp), 1 + ep[1])
        if pose["cached"]:
            kp, vp = wm.attention_qkv(attn, hidden, pose["freqs"], need_q=False)
        else:
            qp, kp, vp = wm.attention_qkv(attn, hidden, pose["freqs"])
            out = wm.sequence_attention(attn, qp, kp, vp, total=pose["total"])
            del qp
            xp = torch.addcmul(xp, attn.o(out.reshape(batch, xp.shape[1], heads * dim)).float(), ep[2])
            xp = xp + self.cross_attn(self.norm3(xp).float(), pose["context"], context_img_len=pose["context_img_len"]).float()
            y = self.ffn(torch.addcmul(ep[3], self.norm2(xp), 1 + ep[4]))
            new_pose = torch.addcmul(xp, y.float(), ep[5])
        if pose["strength"] != 1.0:
            vp = vp * pose["strength"]
    e = (self.modulation.float().unsqueeze(0) + e0.float()).unbind(2)
    e = [tokens.local(m) for m in e]
    q, k, v = wm.attention_qkv(attn, torch.addcmul(e[0], self.norm1(x), 1 + e[1]), freqs)
    strength = plan["ref_strength"] if plan else 1.0
    if strength != 1.0:  # v[:, :hw] — слот изображения-референса (глобальные строки)
        a, _ = plan["rows"]
        hi = max(0, min(n, plan["hw"] - a))
        if hi:
            v = v.clone()
            v[:, :hi] = v[:, :hi] * strength
    if kp is None:
        out = wm.sequence_attention(attn, q, k, v)
    else:
        out = animate2_attention(attn, q, k, v, kp, vp, plan)
    del q, k, v
    x = torch.addcmul(x, attn.o(out.reshape(batch, n, heads * dim)).float(), e[2])
    x = x + self.cross_attn(self.norm3(x).float(), context, context_img_len=context_img_len).float()
    y = self.ffn(torch.addcmul(e[3], self.norm2(x), 1 + e[4]))
    x = torch.addcmul(x, y.float(), e[5])
    tracker = getattr(self, "_ps_tracker", None)
    if tracker is not None:
        tracker.observe(self._ps_finite_slot, x)
    return x, new_pose


class PoseInputCache:
    """Worker-аналог comfy PoseBranchCache: входы pose branch (batch 0, локальные строки) в RAM.

    Ключ — содержимое pose latents (host hash) и id cache-объекта ноды WanAnimate2Cache: новая
    задача / новый cache -> слоты сбрасываются. До 2 слотов (context windows), LRU.
    """

    def __init__(self, ident, dtype):
        self.ident, self.dtype = ident, dtype
        self.slots = OrderedDict()

    def select(self, key, rows):
        slot = self.slots.pop(key, None)
        if slot is None or slot["rows"] != rows:
            slot = dict(rows=rows, blocks={})
        self.slots[key] = slot
        while len(self.slots) > 2:
            self.slots.popitem(last=False)
        return slot

    def store_dtype(self):
        return {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16}.get(str(self.dtype), torch.float16)


def mem_available_bytes():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def cache_fits(slot, local_bytes, world, fraction=0.6):
    """Кэш всех rank (они на одном узле) должен поместиться в долю доступной RAM; уже занятое слотом не считается."""
    available = mem_available_bytes()
    if available is None:
        return True
    held = sum(t.numel() * t.element_size() for t in slot["blocks"].values())
    return (local_bytes - held) * world <= fraction * available


def animate2_forward(net, x, timestep, context, clip_fea, options, extras, cache_spec, time_dim_concat):
    """WanAnimate2Model._forward + forward_orig с sequence sharding обеих ветвей."""
    import comfy.ldm.common_dit as common_dit
    from comfy.ldm.wan.model import sinusoidal_embedding_1d
    wm = _wm()
    if time_dim_concat is not None:
        raise ValueError("Wan Animate2: time_dim_concat не поддерживается native моделью")
    state = net._ps_sequence
    bs, c, t, h, w = x.shape
    patch = net.patch_size
    x = common_dit.pad_to_patch_size(x.float(), patch)
    freqs = net.rope_encode(t, h, w, device=x.device, dtype=torch.float32, transformer_options=options).float()
    pose = extras.get("pose_latents")
    freqs_pose = None
    if pose is not None:
        pose = common_dit.pad_to_patch_size(pose.float(), patch)
        if pose.shape[-2:] != x.shape[-2:]:
            raise ValueError(f"Animate2: pose latents {list(pose.shape[-2:])} != генерация {list(x.shape[-2:])}")
        w_patches = (w + (patch[2] // 2)) // patch[2]
        freqs_pose = net.rope_encode_pose(pose.shape[2], h, w, w_patches, device=x.device, dtype=torch.float32).float()
    xe = net.patch_embedding(x)
    grid_sizes = xe.shape[2:]
    f_gen, gh, gw = grid_sizes
    hw = gh * gw
    tokens = xe.flatten(2).transpose(1, 2)
    del xe
    if pose is not None and pose.shape[2] != f_gen - 1:
        raise ValueError(f"pose branch has {pose.shape[2]} latent frames, expected {f_gen - 1} "
                         "(generation frames minus the reference-image slot)")
    timestep = timestep.float()
    e = net.time_embedding(sinusoidal_embedding_1d(net.freq_dim, timestep.flatten()).float())
    e = e.float().reshape(timestep.shape[0], -1, e.shape[-1])
    e0 = net.time_projection(e).float().unflatten(2, (6, net.dim))
    context_gen = net.text_embedding(context).float()
    context_img_len = None
    if clip_fea is not None:
        if net.img_emb is not None:
            context_gen = torch.cat([net.img_emb(clip_fea.float()).float(), context_gen], dim=1)
        context_img_len = clip_fea.shape[-2]

    total = tokens.shape[1]
    total_p = (f_gen - 1) * hw if pose is not None else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    splittable = all(shard_bounds(n, world - 1, world)[0] < shard_bounds(n, world - 1, world)[1]
                     for n in ((total, total_p) if pose is not None else (total,)))
    state.update(total=total, enabled=world > 1 and splittable, kv_collectives=0, kv_gathered_bytes=0,
                 out_collectives=0, out_exchanged_bytes=0, scale_collectives=0, head_gathers=0)
    a, b = shard_bounds(total, rank, world) if state["enabled"] else (0, total)
    index = wm.TokenIndex(a, b, total, e0.shape[1], tokens.device)
    local = tokens[:, a:b].contiguous()
    freqs_local = freqs[:, a:b]
    del tokens
    ref_strength = extras.get("reference_strength")
    pose_strength = extras.get("pose_strength")
    plan = dict(hw=hw, f_gen=f_gen, total=total, rows=(a, b), pose=None,
                ref_strength=1.0 if ref_strength is None else float(ref_strength))
    slot = None
    cached = False
    if pose is not None:
        ap, bp = shard_bounds(total_p, rank, world) if state["enabled"] else (0, total_p)
        if cache_spec:
            store = getattr(net, "_ps_pose_cache", None)
            if store is None or store.ident != cache_spec["id"]:
                store = net._ps_pose_cache = PoseInputCache(cache_spec["id"], cache_spec.get("dtype"))
            slot = store.select(cache_spec["key"], (ap, bp, total_p, hw))
            cached = len(slot["blocks"]) == len(net.blocks)
            # Хватит ли RAM на кэш всех rank (400+ кадров 960x544: ~14 GB на rank, ~84 GB на 6 GPU)?
            itemsize = torch.empty((), dtype=store.store_dtype()).element_size()
            fits = cached or cache_fits(slot, (bp - ap) * net.dim * len(net.blocks) * itemsize, world)
            if world > 1:  # все rank обязаны идти одной веткой (одинаковые вызовы FSDP units)
                flag = torch.tensor([1 if cached else 0, 1 if fits else 0], device=x.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                cached, fits = bool(flag[0].item()), bool(flag[1].item())
            if not cached:
                slot["blocks"] = {}
            if not fits:
                store.slots.pop(cache_spec["key"], None)
                slot = None
                if not getattr(net, "_ps_pose_cache_warned", False):
                    import warnings
                    warnings.warn("Animate2 cache: не хватает RAM на кэш pose branch всех GPU — шаги идут без кэша "
                                  "(pose branch считается каждый шаг). Уменьшите длину/разрешение или используйте context windows.")
                    net._ps_pose_cache_warned = True
        ones = torch.ones_like(timestep.flatten())
        ep = net.time_embedding(sinusoidal_embedding_1d(net.freq_dim, ones).float()).float()
        ep = ep.reshape(timestep.shape[0], -1, ep.shape[-1])
        e0p = net.time_projection(ep).float().unflatten(2, (6, net.dim))[:, :1]
        local_pose = context_pose = context_img_len_pose = None
        if not cached:
            xp = net.patch_embedding(torch.cat([pose, torch.ones_like(pose[:, :4]), pose], dim=1))
            xp = xp.flatten(2).transpose(1, 2)
            if xp.shape[1] != total_p:
                raise ValueError(f"Animate2 pose tokens {xp.shape[1]} != {total_p}")
            local_pose = xp[:, ap:bp].contiguous()
            del xp
            source = extras.get("context_pose")
            context_pose = net.text_embedding(context if source is None else source).float()
            clip_pose = extras.get("clip_fea_pose")
            clip_pose = clip_fea if clip_pose is None else clip_pose
            if clip_pose is not None:
                if net.img_emb is not None:
                    context_pose = torch.cat([net.img_emb(clip_pose.float()).float(), context_pose], dim=1)
                context_img_len_pose = clip_pose.shape[-2]
        plan["pose"] = dict(e0=e0p, freqs=freqs_pose[:, ap:bp], context=context_pose, cached=cached,
                            context_img_len=context_img_len_pose, total=total_p,
                            strength=1.0 if pose_strength is None else float(pose_strength))
    for i, block in enumerate(net.blocks):
        if plan["pose"] is not None:
            if cached:
                stored = slot["blocks"][i].to(device=local.device, dtype=torch.float32)
                plan["pose"]["x"] = stored.expand(local.shape[0], -1, -1) if stored.shape[0] != local.shape[0] else stored
            else:
                if slot is not None:
                    slot["blocks"][i] = local_pose[:1].to("cpu", dtype=store.store_dtype(), copy=True)
                plan["pose"]["x"] = local_pose
        local, new_pose = block(local, e0, freqs_local, context_gen, context_img_len, index, plan)
        if plan["pose"] is not None and not cached:
            local_pose = new_pose
    out = net.head(local, e, index)
    if state["enabled"]:
        out = wm.gather_tokens(out, total)
        state["head_gathers"] += 1
    return net.unpatchify(out, grid_sizes)[:, :, :t, :h, :w]


# --------------------------------------------------------------- InfiniteTalk
class MultiTalkBlocks(nn.Module):
    """blocks.* из InfiniteTalk/MultiTalk model patch (audio_proj считается на host)."""

    def __init__(self, in_dim, out_dim, num_layers, dtype=None, device=None, operations=None):
        super().__init__()
        from comfy.ldm.wan.model_multitalk import WanMultiTalkAttentionBlock
        self.blocks = nn.ModuleList([WanMultiTalkAttentionBlock(in_dim, out_dim, device=device, dtype=dtype,
                                                                operations=operations) for _ in range(num_layers)])


class MultiTalkEntrypoint(nn.Module):
    """FSDP root model patch: каждый вызов — один блок (внутри блока генератора, после cross-attn)."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, command, index, x, audio, segments, q_pos, k_pos):
        if command != "block":
            raise ValueError(f"Неизвестный вызов MultiTalk: {command}")
        return self.network.blocks[index](x, audio, segments, q_pos, k_pos)


def install_multitalk_compute(net, dispatcher, policy):
    for block in net.blocks:
        block.forward = types.MethodType(multitalk_block_forward, block)
        attn = block.audio_cross_attn
        attn._ps_attention = dispatcher
        attn._ps_safe = policy.fp16_safe
        attn._ps_group = "multitalk_cross"


def rope_1d(x, pos):
    """RotaryPositionalEmbedding1D (MultiTalk): x [..., L, H, D], pos [..., L]; пары (2i, 2i+1)."""
    d = x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device)[:d // 2].float() / d))
    angles = (pos.float()[..., None] * freqs).repeat_interleave(2, dim=-1).unsqueeze(-2)
    x = x.float()
    rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return x * angles.cos() + rotated * angles.sin()


def dynamic_scale(t):
    """q/k без RMSNorm (MultiTalk q_linear/kv_linear): степень двойки по фактическому max."""
    maximum = float(t.abs().amax()) if t.numel() else 0.
    return _wm().power2(maximum * 1.01 / 128.)


def multitalk_block_forward(self, x, audio, segments, q_pos, k_pos):
    """SingleStreamAttention / SingleStreamMultiAttention для локальных строк.

    audio [F_audio, N_a, C]: кадр g латента видит токены кадра g (общие для batch, native batch=1).
    Строки кадров без аудио (x_extra native) получают нулевой residual.
    """
    attn = self.audio_cross_attn
    batch, n, _ = x.shape
    heads, dim = attn.num_heads, attn.head_dim
    q = attn.q_linear(self.norm_x(x)).float().view(batch, n, heads, dim)
    if q_pos is not None:
        q = rope_1d(q, q_pos)
    frames, count = audio.shape[0], audio.shape[1]
    kv = attn.kv_linear(audio.reshape(frames * count, -1)).float().view(frames, count, 2, heads, dim)
    k, v = kv[:, :, 0], kv[:, :, 1]
    if k_pos is not None:
        k = rope_1d(k, k_pos)
    sq, sk = dynamic_scale(q), dynamic_scale(k)
    out = q.new_zeros((batch, n, heads, dim))
    keep = torch.zeros((1, n, 1), device=x.device, dtype=torch.float32)
    for start, stop, g in segments:
        if g >= frames:
            continue
        kk = k[g:g + 1].expand(batch, -1, -1, -1)
        vv = v[g:g + 1].expand(batch, -1, -1, -1)
        out[:, start:stop] = _wm().run_attention(attn, q[:, start:stop], kk, vv, sq, sk, attn._ps_group)
        keep[:, start:stop] = 1.
    return attn.proj(out.reshape(batch, n, heads * dim)).float() * keep


def ref_attention_map(q, k, masks, hw, sq, sk, mode, heads, chunk_elements=1 << 25):
    """get_attn_map_with_target (MultiTalkGetAttnMapPatch) для shard строк -> полная карта [B, C, L] на всех rank.

    Для каждого токена: softmax(q·k_ref) по токенам первого кадра, доля внимания на маску говорящего,
    среднее по heads. mode=None — полные q/k; "token" — локальные q, полные k; "ulysses" — все токены,
    часть heads (сумма + all_reduce).
    """
    batch, length, local_heads, dim = q.shape
    scale = dim ** -.5 * sq * sk
    ref = k[:, :hw].float()
    masks = masks.to(device=q.device, dtype=torch.float32)
    weight = masks / (masks.sum(-1, keepdim=True) + 1e-8)  # [C, hw]
    valid = local_heads
    if mode == "ulysses":
        valid = max(0, min(local_heads, heads - dist.get_rank() * local_heads))
    out = q.new_zeros((batch, masks.shape[0], length), dtype=torch.float32)
    if valid:
        rows = max(1, chunk_elements // max(1, batch * valid * hw))
        for a in range(0, length, rows):
            b = min(length, a + rows)
            logits = torch.einsum("blhd,bmhd->bhlm", q[:, a:b, :valid].float(), ref[:, :, :valid]) * scale
            logits = logits - logits.amax(-1, keepdim=True)
            prob = logits.exp()
            prob = prob / (prob.sum(-1, keepdim=True) + 1e-8)
            out[:, :, a:b] = torch.einsum("bhlm,cm->bcl", prob, weight)
    if mode == "ulysses":
        dist.all_reduce(out)
    elif mode == "token":
        out = _wm().gather_tokens(out.transpose(1, 2).contiguous(), k.shape[1]).transpose(1, 2)
    return out / heads


def speaker_positions(maps, class_interval=4, class_range=24):
    """SingleStreamMultiAttention: позиции RoPE запросов по доминирующему говорящему (2 speakers)."""
    h1, h2 = (0, class_interval), (class_range - class_interval, class_range)

    def scaled(values, target):
        low, high = values.amin(-1, keepdim=True), values.amax(-1, keepdim=True)
        return (values - low) / (high - low + 1e-8) * (target[1] - target[0]) + target[0]
    first, second = scaled(maps[:, 0], h1), scaled(maps[:, 1], h2)
    return torch.where(maps[:, 0] >= maps[:, 1], first, second)  # argmax: при равенстве первый


def encoder_positions(count, device, class_interval=4, class_range=24):
    pos = torch.empty(count, device=device)
    pos[:count // 2] = class_interval / 2
    pos[count // 2:] = class_range - class_interval / 2
    return pos


class MultiTalkStep:
    """attn2_patch InfiniteTalk для локальных строк: x + audio_cross_attn(norm_x(x)) * audio_scale."""

    def __init__(self, holder, audio, rows, hw, scale, state):
        self.root, self.audio, self.scale, self.state = holder.root, audio, scale, state
        self.rows = rows
        self.segments = _wm().frame_segments(rows[0], rows[1], hw)
        self.k_pos = encoder_positions(audio.shape[1], audio.device)

    def __call__(self, index, x):
        maps = self.state.pop("x_ref_attn_map", None)
        q_pos = k_pos = None
        if maps is not None and maps.shape[1] > 1:
            q_pos = speaker_positions(maps)[:, self.rows[0]:self.rows[1]]
            k_pos = self.k_pos
        residual = self.root("block", index, x, self.audio, self.segments, q_pos, k_pos)
        return x + residual.float() * self.scale


def prepare_multitalk(holder, spec, audio, masks, state, total, hw, rows, ref_len):
    if spec is None:
        state["ref_attn_masks"] = None
        return None
    if holder is None:
        raise RuntimeError("InfiniteTalk model patch не загружен в worker")
    if ref_len:
        raise ValueError("InfiniteTalk + reference_latent (ref_conv) не поддерживается: токены сдвинуты")
    if audio is None:
        raise ValueError("InfiniteTalk: нет audio_embeds")
    audio = audio.float()
    if audio.ndim == 4:
        audio = audio[0]  # native: encoder_hidden_states.squeeze(0)
    state["ref_attn_masks"] = masks.float() if masks is not None and masks.shape[0] > 1 else None
    state["ref_attn_hw"] = hw
    state.pop("x_ref_attn_map", None)
    return MultiTalkStep(holder, audio, rows, hw, float(spec.get("audio_scale", 1.0)), state)
