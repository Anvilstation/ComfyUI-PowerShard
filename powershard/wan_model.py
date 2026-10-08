"""Worker-side вычисления Wan поверх родных классов comfy.ldm.wan.

Родная модель (WanModel, VaceWanModel, WanModel_S2V, AnimateWanModel,
CameraWanModel, WanUni3CControlnet) используется для дерева параметров (имена =
checkpoint), rope_encode, unpatchify и вспомогательных энкодеров. Forward блоков и
всех attention-модулей заменяется методами экземпляров: FP32 residual/модуляция/
нормы (как autocast(float32) в оригинальном Wan), FP16 GEMM, FP16 attention через
общий AttentionDispatcher PowerShard.

Sequence parallel: токены режутся по rank до первого блока. Self-attention
(основные, VACE и Uni3C блоки) обменивается token (all-gather K/V) или Ulysses
(all-to-all по heads). Cross-attention к тексту, аудио-инъекции S2V и face-адаптер
Animate покадровые и считаются локально по своим строкам. Каждый FSDP unit
вызывается ровно один раз на rank в одной и той же последовательности, даже если у
rank нет строк нужного вида (иначе разошлись бы all-gather коллективы).
"""
import inspect
import math
import types
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from .config import shard_bounds
from .operations import Linear, RMSNorm


class LayerNorm(nn.LayerNorm):
    def reset_parameters(self):
        pass

    def forward(self, x):
        w = None if self.weight is None else self.weight.float()
        b = None if self.bias is None else self.bias.float()
        return F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps)


class GroupNorm(nn.GroupNorm):
    def reset_parameters(self):
        pass

    def forward(self, x):
        w = None if self.weight is None else self.weight.float()
        b = None if self.bias is None else self.bias.float()
        return F.group_norm(x.float(), self.num_groups, w, b, self.eps)


class Conv1d(nn.Conv1d):
    def reset_parameters(self):
        pass

    def forward(self, x):
        return F.conv1d(x.float(), self.weight.float(), None if self.bias is None else self.bias.float(),
                        self.stride, self.padding, self.dilation, self.groups)


class Conv2d(nn.Conv2d):
    def reset_parameters(self):
        pass

    def forward(self, x):
        return F.conv2d(x.float(), self.weight.float(), None if self.bias is None else self.bias.float(),
                        self.stride, self.padding, self.dilation, self.groups)


class Conv3d(nn.Conv3d):
    def reset_parameters(self):
        pass

    def forward(self, x):
        return F.conv3d(x.float(), self.weight.float(), None if self.bias is None else self.bias.float(),
                        self.stride, self.padding, self.dilation, self.groups)


class Embedding(nn.Embedding):
    def reset_parameters(self):
        pass


class WanOperations:
    Linear = Linear
    RMSNorm = RMSNorm
    LayerNorm = LayerNorm
    GroupNorm = GroupNorm
    Conv1d = Conv1d
    Conv2d = Conv2d
    Conv3d = Conv3d
    Embedding = Embedding


# ------------------------------------------------------------- model classes
def model_class(config):
    """Родной класс comfy по model_type из заголовка checkpoint."""
    import comfy.ldm.wan.model as wan
    kind = config["model_type"]
    if kind == "vace":
        return wan.VaceWanModel
    if kind == "s2v":
        return wan.WanModel_S2V
    if kind == "animate":
        import comfy.ldm.wan.model_animate as animate
        return animate.AnimateWanModel
    if kind in ("camera", "camera_2.2"):
        return wan.CameraWanModel
    if kind == "humo":
        return wan.HumoWanModel
    if kind == "scail":
        return wan.SCAILWanModel
    if kind == "scail2":
        return wan.SCAIL2WanModel
    if kind == "wandancer":
        import comfy.ldm.wan.model_wandancer as dancer
        return dancer.WanDancerModel
    if kind == "animate2":
        import comfy.ldm.wan.model_animate2 as animate2
        return animate2.WanAnimate2Model
    if kind in ("t2v", "i2v"):
        return wan.WanModel
    raise ValueError(f"Wan model_type={kind} не поддерживается PowerShard")


def constructor_kwargs(cls, config):
    """Аргументы конструктора; классы с **kwargs (SCAIL) передают их дальше в WanModel."""
    import comfy.ldm.wan.model as wan
    params = inspect.signature(cls.__init__).parameters
    accepted = set(params)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        accepted |= set(inspect.signature(wan.WanModel.__init__).parameters)
    return {k: v for k, v in config.items() if k in accepted and k not in ("self", "device", "dtype", "operations")}


def build_network(config, dtype=torch.float16, device="meta"):
    """Конструктор родного класса только с поддерживаемыми им аргументами."""
    cls = model_class(config)
    kwargs = constructor_kwargs(cls, config)
    with torch.device(device):
        return cls(**kwargs, dtype=dtype, device=device, operations=WanOperations)


def variant(net):
    if len(net.blocks) and is_kind(net.blocks[0], "WanAnimate2Block"):
        return "animate2"
    if getattr(net, "audio_proj", None) is not None:
        return "humo"
    if getattr(net, "patch_embedding_pose", None) is not None:
        return "scail"
    if getattr(net, "patch_embedding_global", None) is not None:
        return "wandancer"
    if hasattr(net, "vace_blocks"):
        return "vace"
    if hasattr(net, "audio_injector"):
        return "s2v"
    if hasattr(net, "face_adapter"):
        return "animate"
    if getattr(net, "control_adapter", None) is not None:
        return "camera"
    return "base"


# Пути FSDP units помимо блоков (в порядке «листья раньше корня»).
AUX_UNIT_PATHS = ("patch_embedding", "text_embedding", "time_embedding", "time_projection", "img_emb", "ref_conv", "head",
                  "vace_patch_embedding", "control_adapter",
                  "casual_audio_encoder", "cond_encoder", "frame_packer",
                  "pose_patch_embedding", "motion_encoder.enc.net_app", "motion_encoder.enc.fc",
                  "motion_encoder.dec.direction", "face_encoder",
                  "audio_proj",                                                    # HuMo
                  "patch_embedding_pose", "patch_embedding_mask",                  # SCAIL / SCAIL2
                  "patch_embedding_global", "img_emb_refimage", "head_global", "music_projection")  # WanDancer
AUX_UNIT_LISTS = ("vace_blocks", "audio_injector.injector", "audio_injector.injector_adain_layers",
                  "audio_injector.injector_adain_output_layers", "face_adapter.fuser_blocks",
                  "music_encoder", "music_injector.injector")


def submodule(net, path):
    module = net
    for part in path.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def fsdp_unit_modules(net):
    """Блоки + вспомогательные модули, которые оборачиваются отдельными FSDP units."""
    units = list(net.blocks)
    for path in AUX_UNIT_LISTS:
        items = submodule(net, path)
        if items is not None:
            units += list(items)
    for path in AUX_UNIT_PATHS:
        module = submodule(net, path)
        if module is not None and any(True for _ in module.parameters()):
            units.append(module)
    return units


def is_kind(module, name):
    """Проверка по имени класса с учётом FSDP2 (он подменяет класс на подкласс FSDP<Name>)."""
    return any(cls.__name__ == name for cls in type(module).__mro__)


def generated_buffers(root):
    """Buffers, которых нет в checkpoint (вычисляются в __init__ native модулей)."""
    out = {}
    for name, module in root.named_modules():
        if is_kind(module, "Blur") and "kernel" in module._buffers:
            k = torch.tensor([1., 3., 3., 1.])
            kernel = k[None, :] * k[:, None]
            out[(name + ".kernel").removeprefix("network.")] = kernel / kernel.sum()
    return out


# Тензоры, у которых ось 0 — не batch (общие для cond/uncond).
UNBATCHED_KWARGS = ("uni3c_render", "multitalk_audio", "multitalk_masks", "nag_context")


class WanEntrypoint(nn.Module):
    """FSDP root. Всегда через __call__: hooks дочерних units срабатывают."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, command, args, kwargs):
        if command != "forward":
            raise ValueError(f"Неизвестный вызов Wan: {command}")
        x, timestep, context = args
        chunk = getattr(self.network, "_ps_batch_chunk", 0)
        if not chunk or x.shape[0] <= chunk:
            return wan_forward(self.network, x, timestep, context, **kwargs)
        # CFG cond/uncond одним RPC, но последовательными forward: меньше activations,
        # веса собираются FSDP заново для каждой части.
        outputs = []
        for a in range(0, x.shape[0], chunk):
            b = min(a + chunk, x.shape[0])
            part = {k: (v[a:b] if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == x.shape[0]
                        and k not in UNBATCHED_KWARGS else v) for k, v in kwargs.items()}
            part["transformer_options"] = dict(part.get("transformer_options") or {}, _ps_rows=(a, b, x.shape[0]))
            outputs.append(wan_forward(self.network, x[a:b], timestep[a:b] if timestep.ndim and timestep.shape[0] == x.shape[0]
                                       else timestep, context[a:b] if context.shape[0] == x.shape[0] else context, **part))
        return torch.cat(outputs, dim=0)


class Uni3CEntrypoint(nn.Module):
    """FSDP root ControlNet Uni3C: вход и каждый блок — отдельный вызов root."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, command, *args):
        if command == "input":
            return self.network.process_input(*args)
        if command == "block":
            return self.network.forward_block(*args)
        raise ValueError(f"Неизвестный вызов Uni3C: {command}")


def make_fp32_parameters(net):
    """modulation (блоки, VACE блоки, head) хранится в FP32: оригинальный Wan считает её в float32."""
    for module in net.modules():
        p = module._parameters.get("modulation") if hasattr(module, "_parameters") else None
        if isinstance(p, nn.Parameter):
            module.modulation = nn.Parameter(torch.empty(p.shape, dtype=torch.float32, device=p.device), requires_grad=False)


def check_wan_instance(net):
    for name in ("patch_embedding", "text_embedding", "time_embedding", "time_projection", "blocks", "head",
                 "rope_encode", "unpatchify", "freq_dim", "dim", "patch_size"):
        if not hasattr(net, name):
            raise RuntimeError(f"Wan capability: отсутствует {name}")
    for block in list(net.blocks) + list(getattr(net, "vace_blocks", [])):
        for name in ("norm1", "norm2", "norm3", "self_attn", "cross_attn", "ffn", "modulation"):
            if not hasattr(block, name):
                raise RuntimeError(f"Wan block capability: отсутствует {name}")
        for attn in (block.self_attn, block.cross_attn):
            for name in ("q", "k", "v", "o", "norm_q", "norm_k", "num_heads", "head_dim"):
                if not hasattr(attn, name):
                    raise RuntimeError(f"Wan attention capability: отсутствует {name}")
        if not (isinstance(block.ffn, nn.Sequential) and len(block.ffn) == 3):
            raise RuntimeError("Wan FFN: ожидался Sequential[Linear, GELU, Linear]")
    for name in ("norm", "head", "modulation"):
        if not hasattr(net.head, name):
            raise RuntimeError(f"Wan head capability: отсутствует {name}")
    return True


# ----------------------------------------------------------------- sequence
class TokenIndex:
    """Локальные строки [a:b] глобальной последовательности и per-frame модуляция.

    Wan 2.2 TI2V/S2V дают e формы [B, N, ...]; native repeat_e растягивает её на
    токены через repeat_interleave. Здесь тот же индекс i // r только для локальных
    строк, без [B, L, 6, dim] тензора.
    """

    def __init__(self, a, b, total, frames, device):
        self.a, self.b, self.total, self.frames = a, b, total, frames
        self.index = None
        if frames > 1:
            repeats = total // frames
            if repeats * frames != total:
                repeats += 1
            if repeats < 1:
                raise ValueError("Число токенов меньше числа timestep frames")
            self.index = torch.arange(a, b, device=device) // repeats

    def local(self, e):
        if self.index is None or e.shape[1] == 1:
            return e
        return e.index_select(1, self.index)


def frame_segments(a, b, group):
    """[(local_start, local_stop, group_index)] для глобальных строк [a,b), группы по `group` токенов."""
    out, row = [], a
    while row < b:
        g = row // group
        stop = min(b, (g + 1) * group)
        out.append((row - a, stop - a, g))
        row = stop
    return out


def rope_apply(x, freqs):
    """[B,L,H,D] x [1,L,1,D/2,2,2] -> FP32; то же, что comfy _apply_rope1."""
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape)


def power2(value):
    return 2. ** max(0, math.ceil(math.log2(max(1., value))))


def gather_tokens(x, total, dtype=None):
    """Batch-first all-gather по оси токенов: [B,n,...] -> [B,total,...]."""
    from .attention import gather_rows
    rows = x.transpose(0, 1).contiguous()
    return gather_rows(rows, total, dtype=dtype).transpose(0, 1).contiguous()


def ulysses_to_sequence(tensors, total, heads):
    """Local tokens/all heads -> all tokens/head shard; batch-first [B,n,H,D].

    Heads дополняются нулями до ceil(H/W)*W, токены до ceil(L/W)*W. Паддинг
    токенов стоит только в хвосте (shard_bounds), поэтому [:total] его снимает.
    """
    world = dist.get_world_size()
    width, h = math.ceil(total / world), math.ceil(heads / world)
    a, b = shard_bounds(total, dist.get_rank(), world)
    batch, n, _, dim = tensors[0].shape
    if n != b - a or any(t.shape != (batch, n, heads, dim) for t in tensors):
        raise ValueError("Ulysses local token/head geometry mismatch")
    parts = []
    for t in tensors:
        padded = t.new_zeros((batch, width, h * world, dim))
        padded[:, :n, :heads].copy_(t)
        parts.append(padded.view(batch, width, world, h, dim).permute(2, 0, 1, 3, 4))
    send = torch.stack(parts, dim=1).contiguous()  # [W(dst), T, B, width, h, D]
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)  # recv[s] = токены rank s, мои heads
    full = recv.permute(1, 2, 0, 3, 4, 5).reshape(len(tensors), batch, world * width, h, dim)
    return tuple(t[:, :total] for t in full), h * world, send.numel() * send.element_size()


def ulysses_to_heads(out, total, heads):
    """Inverse: [B,total,h,D] -> [B,n,H,D]; удаляет фиктивные heads."""
    world = dist.get_world_size()
    width = math.ceil(total / world)
    batch, _, h, dim = out.shape
    if out.shape[1] != total or h != math.ceil(heads / world):
        raise ValueError("Ulysses global token/head geometry mismatch")
    padded = out.new_zeros((batch, world * width, h, dim))
    padded[:, :total].copy_(out)
    send = padded.view(batch, world, width, h, dim).permute(1, 0, 2, 3, 4).contiguous()  # [W(dst),B,width,h,D]
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)  # recv[s] = heads rank s для моих токенов
    a, b = shard_bounds(total, dist.get_rank(), world)
    result = recv.permute(1, 2, 0, 3, 4).reshape(batch, width, world * h, dim)[:, :b - a, :heads]
    return result.contiguous(), send.numel() * send.element_size()


# ---------------------------------------------------------------- attention
def scaled_half(x, scale):
    return (x.float() / scale).half()


def v_scale(v, collective):
    """Степень двойки по max|V|; при collective — общий MAX для согласованного wire."""
    from .fp16_safe import power2_scale
    maximum = v.float().abs().amax() if v.numel() else v.new_zeros((), dtype=torch.float32)
    if collective:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return power2_scale(maximum / 16384.)


def run_attention(module, q, k, v, sq, sk, group, sv=None, restore_v=True):
    """q,k,v BLHD; softmax scale учитывает статические sq/sk.

    sv=None: q/k ещё не нормализованы (FP32/FP16 любые) — делятся здесь, V
    масштабируется по своему max. sv задан: q/k/v уже нормализованы до обмена
    (FP16 wire), V в единицах sv. restore_v=False возвращает FP16 в единицах sv.
    """
    from .attention_contract import AttentionOptions
    if q.shape[1] == 0:
        empty = q.new_zeros(q.shape[:-1] + (v.shape[-1],), dtype=torch.float32)
        return empty if restore_v else empty.half()
    if sv is None:
        q16, k16 = scaled_half(q, sq), scaled_half(k, sk)
        sv = v_scale(v, False)
        v16 = scaled_half(v, sv)
    else:
        q16, k16, v16 = q.half(), k.half(), v.half()
    scale = q.shape[-1] ** -.5 * sq * sk
    out = module._ps_attention(q16.contiguous(), k16.contiguous(), v16.contiguous(),
                               AttentionOptions(softmax_scale=scale), group=group, compute_fp16=module._ps_safe)
    return out.float() * sv if restore_v else out


def attention_qkv(self, x, freqs, need_q=True):
    """WanSelfAttention q/k (RMSNorm + RoPE, FP32) и v: [B,n,H,D]."""
    batch, n, _ = x.shape
    heads, dim = self.num_heads, self.head_dim
    k = rope_apply(self.norm_k(self.k(x)).view(batch, n, heads, dim), freqs)
    v = self.v(x).view(batch, n, heads, dim)
    if not need_q:
        return k, v
    q = rope_apply(self.norm_q(self.q(x)).view(batch, n, heads, dim), freqs)
    return q, k, v


def self_attention_forward(self, x, freqs, transformer_options=None):
    """x [B,n,dim] (FP32) -> [B,n,dim]. Distributed по состоянию _ps_sequence."""
    batch, n, _ = x.shape
    q, k, v = attention_qkv(self, x, freqs)
    out = sequence_attention(self, q, k, v)
    return self.o(out.reshape(batch, n, self.num_heads * self.head_dim))


def sequence_attention(self, q, k, v, total=None):
    """Полная self-attention для локальных строк (token all-gather K/V или Ulysses all-to-all).

    total — длина глобальной последовательности (по умолчанию основной; Animate2 pose — своя).
    InfiniteTalk 2 speakers: в блоках основной сети здесь же считается x_ref_attn_map.
    """
    from .telemetry import region
    state = self._ps_sequence
    heads = self.num_heads
    sq, sk = self._ps_qk_scales[:2]
    # Карта говорящих только для основной последовательности блоков генератора (не pose branch Animate2).
    ref_masks = state.get("ref_attn_masks") if getattr(self, "_ps_main", False) and total is None else None
    if not state["enabled"]:
        if ref_masks is not None:
            from .wan_extras import ref_attention_map
            state["x_ref_attn_map"] = ref_attention_map(q, k, ref_masks, state["ref_attn_hw"], 1., 1., None, heads)
        out = run_attention(self, q, k, v, sq, sk, self._ps_group)
    else:
        total = total or state["total"]
        fp16_wire = state["comm_dtype"] == "fp16"
        sv = None
        if fp16_wire:
            # Нормализация ДО обмена: q/k статические (RMSNorm+RoPE bounds), V общий MAX.
            q, k = scaled_half(q, sq), scaled_half(k, sk)
            sv = v_scale(v, True)
            state["scale_collectives"] += 1
            v = scaled_half(v, sv)
        else:
            q, k, v = q.float(), k.float(), v.float()
        if state["ulysses"]:
            with region("wan_ulysses_QKV_all_to_all"):
                (q, k, v), padded, exchanged = ulysses_to_sequence((q, k, v), total, heads)
            state["padded_heads"] = padded
            state["kv_collectives"] += 1
            state["kv_gathered_bytes"] += exchanged
            if ref_masks is not None:
                from .wan_extras import ref_attention_map
                scale = (sq, sk) if fp16_wire else (1., 1.)
                state["x_ref_attn_map"] = ref_attention_map(q, k, ref_masks, state["ref_attn_hw"], *scale,
                                                            "ulysses", heads)
            # FP16 wire: обратный обмен в нормализованных единицах, восстановление после.
            out = run_attention(self, q, k, v, sq, sk, self._ps_group, sv=sv, restore_v=not fp16_wire)
            with region("wan_ulysses_output_all_to_all"):
                out, exchanged = ulysses_to_heads(out, total, heads)
            state["out_collectives"] += 1
            state["out_exchanged_bytes"] += exchanged
            if fp16_wire:
                out = out.float() * sv
        else:
            with region("wan_sequence_KV_all_gather"):
                kv = gather_tokens(torch.stack((k, v), dim=2), total)
            state["kv_collectives"] += 1
            state["kv_gathered_bytes"] += kv.numel() * kv.element_size()
            k, v = kv.unbind(2)
            if ref_masks is not None:
                from .wan_extras import ref_attention_map
                scale = (sq, sk) if fp16_wire else (1., 1.)
                state["x_ref_attn_map"] = ref_attention_map(q, k, ref_masks, state["ref_attn_hw"], *scale,
                                                            "token", heads)
            out = run_attention(self, q, k, v, sq, sk, self._ps_group, sv=sv)
        state["effective_comm_dtype"] = "float16" if fp16_wire else "float32"
    return out


def cross_attention_forward(self, x, context, context_img_len=None, transformer_options=None, frame_index=None, **kwargs):
    """Запрос — локальные токены; K/V — текст (и CLIP image) или покадровый контекст.

    frame_index (S2V аудио-инъекция): context [B, F, N, C] или [B, F, C], строка
    x[:, i] видит только context кадра frame_index[i] (native rearrange "(b t) n c").
    """
    batch, n, _ = x.shape
    heads, dim = self.num_heads, self.head_dim
    scales = self._ps_qk_scales
    q = self.norm_q(self.q(x)).view(batch, n, heads, dim)
    if frame_index is not None:
        if context.ndim == 3:
            context = context.unsqueeze(2)
        frames, tokens = context.shape[1], context.shape[2]
        flat = context.reshape(batch, frames * tokens, -1)
        k = self.norm_k(self.k(flat)).view(batch, frames, tokens, heads, dim)
        v = self.v(flat).view(batch, frames, tokens, heads, dim)
        out = x.new_zeros((batch, n, heads, dim), dtype=torch.float32)
        for start, stop, g in frame_index:
            out[:, start:stop] = run_attention(self, q[:, start:stop], k[:, g], v[:, g], scales[0], scales[1], self._ps_group)
        return self.o(out.reshape(batch, n, heads * dim))
    # Native WanI2VCrossAttention: context[:, :None] / [None:] — при context_img_len=None image-ветвь
    # видит весь контекст (I2V-модель без CLIP: Animate2/InfiniteTalk); 0 (WanDancer без CLIP) — пустая ветвь.
    img = getattr(self, "k_img", None) is not None and context_img_len != 0
    if img:
        if context_img_len is None:
            context_img = context
        else:
            context_img, context = context[:, :context_img_len], context[:, context_img_len:]
    k = self.norm_k(self.k(context)).view(batch, context.shape[1], heads, dim)
    v = self.v(context).view(batch, context.shape[1], heads, dim)
    out = run_attention(self, q, k, v, scales[0], scales[1], self._ps_group)
    nag = self._ps_sequence.get("nag") if getattr(self, "_ps_nag", False) else None
    if nag is not None and nag["rows"]:
        from .accel import nag_combine
        count = len(nag["rows"])
        rows = torch.tensor(nag["rows"], device=x.device, dtype=torch.long)
        negative = nag["context"]
        kn = self.norm_k(self.k(negative)).view(negative.shape[0], -1, heads, dim).expand(count, -1, -1, -1)
        vn = self.v(negative).view(negative.shape[0], -1, heads, dim).expand(count, -1, -1, -1)
        out_neg = run_attention(self, q.index_select(0, rows), kn, vn, scales[0], scales[1], self._ps_group + "_nag")
        mixed = nag_combine(out.index_select(0, rows).reshape(count, n, -1), out_neg.reshape(count, n, -1),
                            nag["scale"], nag["tau"], nag["alpha"])
        out = out.index_copy(0, rows, mixed.view(count, n, heads, dim).to(out.dtype))
    if img:
        ki = self.norm_k_img(self.k_img(context_img)).view(batch, context_img.shape[1], heads, dim)
        vi = self.v_img(context_img).view(batch, context_img.shape[1], heads, dim)
        out = out + run_attention(self, q, ki, vi, scales[0], scales[2], self._ps_group + "_img")
    return self.o(out.reshape(batch, n, heads * dim))


def face_block_forward(self, x, motion_vec, rows):
    """Animate FaceBlock: строки кадра g видят motion tokens кадра g (native "(B L) S")."""
    a, b, total = rows
    batch, frames, tokens, channels = motion_vec.shape
    group = total // frames
    if group * frames != total:
        raise ValueError(f"Animate face adapter: {total} токенов не делятся на {frames} кадров")
    heads = self.heads_num
    dim = channels // heads
    kv = self.linear1_kv(self.pre_norm_motion(motion_vec))
    k, v = kv.float().view(batch, frames, tokens, 2, heads, dim).unbind(3)
    k = self.k_norm(k)
    q = self.q_norm(self.linear1_q(self.pre_norm_feat(x)).float().view(batch, x.shape[1], heads, dim))
    out = x.new_zeros((batch, x.shape[1], heads, dim), dtype=torch.float32)
    sq, sk = self._ps_qk_scales[:2]
    for start, stop, g in frame_segments(a, b, group):
        out[:, start:stop] = run_attention(self, q[:, start:stop], k[:, g], v[:, g], sq, sk, "wan_face")
    return self.linear2(out.reshape(batch, x.shape[1], heads * dim))


def adain_forward(self, x, temb, frame_index):
    """S2V AdaLayerNorm с per-frame temb [B,F,C]: одна FSDP-коллективная операция на вызов."""
    shift, scale = self.linear(self.silu(temb)).float().chunk(2, dim=-1)
    index = torch.cat([torch.full((stop - start,), g, device=x.device, dtype=torch.long) for start, stop, g in frame_index]) \
        if frame_index else torch.zeros((0,), device=x.device, dtype=torch.long)
    return self.norm(x) * (1 + scale.index_select(1, index)) + shift.index_select(1, index)


def ffn_forward(self, x):
    """Linear -> GELU(tanh) -> Linear по token chunks; политика chunk как у H3 MLP."""
    return chunked_mlp(self, x, self[0], self[2])


def chunked_mlp(self, x, fc1, fc2, region_name="Wan_FFN_chunks"):
    """Общая реализация chunked MLP (Wan ffn, LTX FeedForward): self — модуль с политикой/контекстом памяти."""
    from .memory_policy import mlp_plan, mlp_budget
    from .operations import prepared_linears
    from .telemetry import region
    shape = x.shape
    rows = x.reshape(-1, shape[-1])
    policy = self._ps_mlp_policy
    context = getattr(self, "_ps_memory_context", {})
    mode = policy.mlp_chunk_mode
    output_bytes = rows.shape[0] * fc2.out_features * 4
    if mode == "off" and "mlp_budget_bytes" in context:
        # off = «целиком, если помещается»: длинные видео (Animate2 400+ кадров -> 37k строк/GPU, ~10 GB
        # workspace) иначе дают CUDA OOM. Числа и порядок суммирования в chunk не меняются.
        full = mlp_plan(len(rows), shape[-1], fc1.out_features, "off", policy.mlp_chunk_tokens)
        if full["estimated_chunk_workspace_bytes"] + output_bytes > context["mlp_budget_bytes"]:
            mode = "auto"
    if mode == "auto":
        budget, info = mlp_budget(context, x.device)
        if policy.mlp_chunk_mode == "off":
            info = dict(info, override="off -> auto: полный локальный MLP не помещается в оценку памяти")
    else:
        budget = context.get("mlp_budget_bytes", 256 * 2**20)
        info = dict(budget_source="RPC boundary; no per-block allocator sampling")
    plan = mlp_plan(len(rows), shape[-1], fc1.out_features, mode, policy.mlp_chunk_tokens,
                    max(0, budget - output_bytes))
    plan.update(info)
    self._ps_mlp_report = plan
    if context.get("_progress"):
        context["_progress"]("mlp_plan", dict(module=self._ps_label, **plan))
    result = torch.empty((rows.shape[0], fc2.out_features), dtype=torch.float32, device=x.device)
    safe = getattr(fc1, "_ps_safe", False) and not getattr(fc1, "_ps_fp32", False)
    allowance = max(0, budget - plan["estimated_chunk_workspace_bytes"] - output_bytes)
    with region(region_name), prepared_linears((fc1, fc2), allowance, enabled=safe and plan["chunks"] > 1) as prep:
        plan.update(prep)
        for a in range(0, rows.shape[0], plan["effective_tokens"]):
            b = min(a + plan["effective_tokens"], rows.shape[0])
            hidden = F.gelu(fc1(rows[a:b]).float(), approximate="tanh")
            result[a:b] = fc2(hidden).float()
            del hidden
    return result.reshape(shape[:-1] + (fc2.out_features,))


def block_body(self, x, e0, freqs, context, context_img_len, tokens, hooks=None):
    """WanAttentionBlock в FP32 residual. e0: [B,N,6,dim] FP32 (глобальный по timestep).

    hooks (между cross-attention и FFN, внутри того же FSDP unit):
      humo=(audio [B,F,16,C], frame segments) — HuMo audio_cross_attn_wrapper;
      multitalk=callable(block_index, x) — InfiniteTalk attn2_patch (своя FSDP root).
    """
    e = (self.modulation.float().unsqueeze(0) + e0.float()).unbind(2)
    e = [tokens.local(m) for m in e]
    y = self.self_attn(torch.addcmul(e[0], self.norm1(x), 1 + e[1]), freqs)
    x = torch.addcmul(x, y.float(), e[2])
    x = x + self.cross_attn(self.norm3(x).float(), context, context_img_len=context_img_len).float()
    if hooks:
        x = after_cross_hooks(self, x, hooks)
    y = self.ffn(torch.addcmul(e[3], self.norm2(x), 1 + e[4]))
    x = torch.addcmul(x, y.float(), e[5])
    tracker = getattr(self, "_ps_tracker", None)
    if tracker is not None:
        tracker.observe(self._ps_finite_slot, x)
    return x


def after_cross_hooks(self, x, hooks):
    audio = hooks.get("humo")
    wrapper = getattr(self, "audio_cross_attn_wrapper", None)
    if audio is not None and wrapper is not None:
        x = x + wrapper.audio_cross_attn(wrapper.norm1_audio(x).float(), audio[0], frame_index=audio[1]).float()
    talk = hooks.get("multitalk")
    if talk is not None:
        x = talk(self._ps_block_index, x)
    return x


def block_forward(self, x, e0, freqs, context, context_img_len, tokens, hooks=None):
    return block_body(self, x, e0, freqs, context, context_img_len, tokens, hooks)


def vace_block_forward(self, c, x, e0, freqs, context, tokens):
    """VaceWanAttentionBlock: before_proj (block 0) + блок + after_proj, токены как у x."""
    if self.block_id == 0:
        c = self.before_proj(c).float() + x
    c = block_body(self, c, e0, freqs, context, None, tokens)
    return self.after_proj(c).float(), c


def head_forward(self, x, e, tokens):  # Head и WanDancer head_global
    mod = (self.modulation.float().unsqueeze(0) + e.float().unsqueeze(2)).unbind(2)
    shift, scale = tokens.local(mod[0]), tokens.local(mod[1])
    return self.head(torch.addcmul(shift, self.norm(x), 1 + scale)).float()


# ------------------------------------------------------------- installation
def attention_modules(net):
    """name -> (module, kind, norms, rope) для всех attention с RMSNorm q/k."""
    import comfy.ldm.wan.model as wan
    cross_types = tuple(t for t in (getattr(wan, "WanT2VCrossAttention", None), getattr(wan, "WanI2VCrossAttention", None)) if t)
    result = {}
    for name, module in net.named_modules():
        if cross_types and isinstance(module, cross_types):
            norms = ("norm_q", "norm_k") + (("norm_k_img",) if getattr(module, "k_img", None) is not None else ())
            result[name] = (module, "cross", norms, False)
        elif is_kind(module, "WanT2VCrossAttentionGather"):  # HuMo: покадровое аудио (16 токенов/кадр)
            result[name] = (module, "cross", ("norm_q", "norm_k"), False)
        elif isinstance(module, wan.WanSelfAttention):
            result[name] = (module, "self", ("norm_q", "norm_k"), True)
        elif is_kind(module, "FaceBlock"):
            result[name] = (module, "face", ("q_norm", "k_norm"), False)
    return result


def configure_network(net, config, options, dispatcher):
    """Общая подготовка (worker и тесты): tracker, FP16/FP32 Linear политика, forwards.

    Вызывается на meta-модели ДО FSDP и загрузки весов. Возвращает (tracker, state).
    """
    from .fp16_safe import FiniteTracker, install_safe_operations
    tracker = FiniteTracker(options.debug_finite)
    install_safe_operations(net, tracker, options.fp16_safe)
    for name, module in net.named_modules():
        if name.startswith(FP32_LINEAR_PREFIXES) and hasattr(module, "_ps_fp32"):
            module._ps_safe, module._ps_fp32 = True, True  # маленькие GEMM в FP32, как autocast(float32)
    net._ps_batch_chunk = options.batch_chunk
    state = install_wan_compute(net, dispatcher, config, options)
    return tracker, state


FP32_LINEAR_PREFIXES = ("time_embedding.", "time_projection.", "img_emb.", "img_emb_refimage.", "audio_proj.",
                        "music_projection", "music_encoder.")


def new_sequence_state(config):
    return dict(total=0, enabled=False, ulysses=config.sequence_mode == "ulysses",
                comm_dtype=config.sequence_comm_dtype, mode=config.sequence_mode)


def install_wan_compute(net, dispatcher, config, policy, state=None):
    """Методы экземпляров вместо native forward; без глобальных monkey patches.

    Работает для WanModel и подклассов (VACE/S2V/Animate/Camera) и для Uni3C.
    """
    import comfy.ldm.wan.model as wan
    state = state or new_sequence_state(config)
    net._ps_sequence = state
    from .wan_extras import music_attention_forward, animate2_block_forward
    for name, module in net.named_modules():
        if is_kind(module, "MusicSelfAttention"):  # WanDancer music encoder: крошечная, FP32 SDPA
            module.forward = types.MethodType(music_attention_forward, module)
    for name, (module, kind, _, _) in attention_modules(net).items():
        method = {"self": self_attention_forward, "cross": cross_attention_forward, "face": face_block_forward}[kind]
        module.forward = types.MethodType(method, module)
        module._ps_attention = dispatcher
        module._ps_safe = policy.fp16_safe
        module._ps_sequence = state
        module._ps_group = "wan_" + kind if not name.startswith("controlnet_blocks") else "uni3c_" + kind
        module._ps_qk_scales = (1., 1., 1.)
    for name, module in net.named_modules():
        if isinstance(module, nn.Sequential) and len(module) == 3 and isinstance(module[1], nn.GELU) and name.endswith("ffn"):
            module.forward = types.MethodType(ffn_forward, module)
            module._ps_mlp_policy = policy
            module._ps_label = name
        if is_kind(module, "AdaLayerNorm"):
            module.forward = types.MethodType(adain_forward, module)
    if hasattr(net, "blocks") and isinstance(getattr(net, "head", None), wan.Head):
        check_wan_instance(net)
        for index, block in enumerate(net.blocks):
            method = animate2_block_forward if is_kind(block, "WanAnimate2Block") else block_forward
            block.forward = types.MethodType(method, block)
            block._ps_block_index = index
            block.self_attn._ps_main = True  # x_ref_attn_map (InfiniteTalk) только в блоках генератора
            block.cross_attn._ps_nag = True   # NAG — только текстовая cross-attention блоков генератора
        for block in getattr(net, "vace_blocks", []):
            block.forward = types.MethodType(vace_block_forward, block)
        for head in (net.head, getattr(net, "head_global", None)):
            if head is not None:
                head.forward = types.MethodType(head_forward, head)
        net._ps_variant = variant(net)
    return state


def qk_norm_names(net):
    """Имена norm весов, ограничивающих q/k (для статических power-of-two scales)."""
    names = {}
    for name, (module, kind, norms, rope) in attention_modules(net).items():
        for norm in norms:
            sub = getattr(module, norm, None)
            if sub is not None and getattr(sub, "weight", None) is not None:
                names[f"{name}.{norm}.weight"] = (name, norm)
    return names


def apply_qk_scales(net, maxima):
    """|RMSNorm(x)_j| <= sqrt(n)*max|w| (n — нормируемая длина); RoPE добавляет sqrt(2). Цель |q/s|<=128."""
    for name, (module, kind, norms, rope) in attention_modules(net).items():
        values = []
        for norm in norms:
            sub = getattr(module, norm)
            maximum = maxima.get(f"{name}.{norm}.weight", 1.)
            if not math.isfinite(maximum):
                raise FloatingPointError(f"Non-finite {name}.{norm} weights")
            width = sub.weight.shape[0] if getattr(sub, "weight", None) is not None else module.head_dim
            bound = (math.sqrt(2.) if rope else 1.) * math.sqrt(width) * maximum * 1.01
            values.append(power2(bound / 128.))
        while len(values) < 3:
            values.append(1.)
        module._ps_qk_scales = tuple(values)


# ------------------------------------------------------------------ forward
SUPPORTED_KWARGS = {
    "base": (), "camera": ("camera_conditions",),
    "vace": ("vace_context", "vace_strength"),
    "s2v": ("audio_embed", "reference_motion", "control_video"),
    "animate": ("pose_latents", "face_pixel_values"),
    "humo": ("audio_embed",),
    "scail": ("pose_latents", "ref_mask_latents", "sam_latents", "ref_mask_flag"),
    "wandancer": ("audio_embed", "clip_fea_ref", "fps", "audio_inject_scale"),
    "animate2": ("pose_latents", "clip_fea_pose", "context_pose", "pose_strength", "reference_strength"),
}


def wan_forward(net, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options=None,
                reference_latent=None, _powershard_rope_options=None, _powershard_uni3c=None, uni3c_render=None,
                _powershard_uni3c_runtime=None, _powershard_multitalk=None, multitalk_audio=None, multitalk_masks=None,
                _powershard_multitalk_runtime=None, _powershard_animate2_cache=None, _powershard_block_cache=None,
                _powershard_nag=None, nag_context=None, _powershard_riflex=None, **kwargs):
    """Эквивалент WanModel._forward + forward_orig (и подклассов) с sequence sharding."""
    import comfy.ldm.common_dit
    from comfy.ldm.wan.model import sinusoidal_embedding_1d
    from .wan_extras import animate2_forward, music_embedding, prepare_multitalk
    kind = getattr(net, "_ps_variant", "base")
    extras = {k: v for k, v in kwargs.items() if v is not None}
    unsupported = sorted(set(extras) - set(SUPPORTED_KWARGS[kind]))
    if unsupported:
        raise ValueError(f"PowerShard Wan ({kind}): conditioning {unsupported} не поддерживается этой моделью")
    if _powershard_uni3c is not None and kind not in ("base", "camera"):
        raise ValueError(f"Uni3C ControlNet с Wan {kind} не поддерживается (сдвинутые/дополнительные токены)")
    if _powershard_multitalk is not None and kind not in ("base", "camera"):
        raise ValueError(f"InfiniteTalk с Wan {kind} не поддерживается: нужен Wan 2.1 I2V/T2V 14B")
    options = dict(transformer_options or {})
    if _powershard_rope_options:
        options["rope_options"] = dict(_powershard_rope_options)
    state = net._ps_sequence
    state["ref_attn_masks"] = None
    state["nag"] = None
    state["blocks_skipped"] = False
    if kind == "animate2" and (_powershard_block_cache or _powershard_nag or _powershard_riflex):
        raise ValueError("Block Cache / NAG / RIFLEx для Wan Animate2 не поддерживаются (отдельный pose-branch forward)")
    state.pop("x_ref_attn_map", None)
    if kind == "animate2":
        return animate2_forward(net, x, timestep, context, clip_fea, options, extras, _powershard_animate2_cache,
                                time_dim_concat)
    bs, c, t, h, w = x.shape
    pad = lambda v: comfy.ldm.common_dit.pad_to_patch_size(v.float(), net.patch_size)  # noqa: E731
    x = pad(x)
    t_len = t
    if time_dim_concat is not None:
        time_dim_concat = pad(time_dim_concat)
        x = torch.cat([x, time_dim_concat], dim=2)
        t_len = x.shape[2]
    use_ref = getattr(net, "ref_conv", None) is not None and reference_latent is not None
    fps = 30.
    ref_frames = 0  # SCAIL: кадры референса перед видео по времени (срезаются после unpatchify)
    pose = ref_mask = sam = None
    if kind == "scail":
        pose = None if extras.get("pose_latents") is None else pad(extras["pose_latents"])
        ref_mask = None if extras.get("ref_mask_latents") is None else pad(extras["ref_mask_latents"])
        sam = None if extras.get("sam_latents") is None else pad(extras["sam_latents"])
        if (ref_mask is not None or sam is not None) and getattr(net, "patch_embedding_mask", None) is None:
            raise ValueError("SCAIL: маски (ref_mask/sam) есть только у SCAIL-2 checkpoint")
        if reference_latent is not None:
            reference_latent = pad(reference_latent)
            t_len += reference_latent.shape[2]
        freqs = net.rope_encode(t_len, h, w, device=x.device, dtype=torch.float32, transformer_options=options,
                                pose_latents=pose, reference_latent=reference_latent,
                                ref_mask_flag=extras.get("ref_mask_flag"))
        use_ref = False
    elif kind == "wandancer":
        if reference_latent is not None:
            raise ValueError("WanDancer: reference_latent (ref_conv) не используется оригинальным pipeline "
                             "и несовместим с native RoPE")
        fps = float(extras.get("fps", 30.))
        freqs = net.rope_encode(t_len, h, w, fps=fps, device=x.device, dtype=torch.float32, transformer_options=options)
        use_ref = False
    else:
        if use_ref:
            t_len += 1
        with riflex_context(net, _powershard_riflex, t_len):
            freqs = net.rope_encode(t_len, h, w, device=x.device, dtype=torch.float32, transformer_options=options)
    freqs = freqs.float()
    x_input = x
    timestep = timestep.float()

    # --- patch embedding и вариант-специфичные входы (реплицированно, дёшево) ---
    audio_emb = audio_emb_global = motion_vec = None
    if kind == "s2v" and extras.get("audio_embed") is not None:
        num_embeds = x.shape[-3] * 4
        audio_emb_global, audio_emb = net.casual_audio_encoder(extras["audio_embed"].float()[:, :, :, :num_embeds])
    if kind == "scail" and reference_latent is not None:
        x = torch.cat((reference_latent, x), dim=2)
        ref_frames = reference_latent.shape[2]
    global_fps = kind == "wandancer" and int(fps + 0.5) != 30
    xe = (net.patch_embedding_global if global_fps else net.patch_embedding)(x)
    if ref_mask is not None:
        xe = xe + net.patch_embedding_mask(ref_mask).float()
    if kind == "camera" and extras.get("camera_conditions") is not None:
        xe = xe + net.control_adapter(extras["camera_conditions"].float()).float()
    if kind == "s2v" and extras.get("control_video") is not None:
        xe = xe + net.cond_encoder(extras["control_video"].float()).float()
    if kind == "animate":
        face = extras.get("face_pixel_values")
        xe, motion_vec = net.after_patch_embedding(xe, extras.get("pose_latents"),
                                                   None if face is None else face.half())
        xe = xe.float()
    if kind == "s2v" and timestep.ndim == 1:
        timestep = timestep.unsqueeze(1).repeat(1, xe.shape[2])
    grid_sizes = xe.shape[2:]
    tokens = xe.flatten(2).transpose(1, 2)
    seq_len = tokens.shape[1]
    del xe
    if pose is not None:  # SCAIL pose tokens в хвосте последовательности
        pe = net.patch_embedding_pose(pose).float()
        if sam is not None:
            pe = pe + net.patch_embedding_mask(sam).float()
        tokens = torch.cat([tokens, pe.flatten(2).transpose(1, 2)], dim=1)
        del pe
    humo_frames = None
    if kind == "humo" and reference_latent is not None:
        ref = net.patch_embedding(reference_latent.float()).flatten(2).transpose(1, 2)
        freqs_ref = net.rope_encode(reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1],
                                    t_start=x.shape[2], device=x.device, dtype=torch.float32).float()
        tokens = torch.cat([tokens, ref], dim=1)
        freqs = torch.cat([freqs, freqs_ref], dim=1)
        del ref, freqs_ref
    if kind == "s2v":
        mask = net.trainable_cond_mask.weight.float().unsqueeze(1).unsqueeze(1)
        tokens = tokens + mask[0]
        if reference_latent is not None:
            ref = net.patch_embedding(reference_latent.float()).flatten(2).transpose(1, 2)
            freqs_ref = net.rope_encode(reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1],
                                        t_start=max(30, x.shape[2] + 9), device=x.device, dtype=torch.float32).float()
            tokens = torch.cat([tokens, ref + mask[1]], dim=1)
            freqs = torch.cat([freqs, freqs_ref], dim=1)
            timestep = torch.cat([timestep, timestep.new_zeros((timestep.shape[0], reference_latent.shape[-3]))], dim=1)
        if extras.get("reference_motion") is not None:
            motion, freqs_motion = net.frame_packer(extras["reference_motion"].float(), net)
            tokens = torch.cat([tokens, motion.float() + mask[2]], dim=1)
            freqs = torch.cat([freqs, freqs_motion.float()], dim=1)
            timestep = torch.repeat_interleave(timestep, 2, dim=1)
            timestep = torch.cat([timestep, timestep.new_zeros((timestep.shape[0], 3))], dim=1)
    e = net.time_embedding(sinusoidal_embedding_1d(net.freq_dim, timestep.flatten()).float())
    e = e.float().reshape(timestep.shape[0], -1, e.shape[-1])
    e0 = net.time_projection(e).float().unflatten(2, (6, net.dim))
    ref_len = 0
    if use_ref:
        ref = net.ref_conv(reference_latent.float()).flatten(2).transpose(1, 2)
        tokens = torch.cat((ref, tokens), dim=1)
        ref_len = ref.shape[1]
    context = net.text_embedding(context).float()
    if _powershard_nag and nag_context is not None:
        from .accel import nag_rows
        state["nag"] = dict(_powershard_nag, context=net.text_embedding(nag_context.float()).float(),
                            rows=batch_rows(nag_rows(options.get("cond_or_uncond"), batch_total(options, bs)), options))
    context_img_len = None
    if kind == "wandancer":
        context_img_len = 0
        if net.img_emb is not None and clip_fea is not None:
            context = torch.cat([net.img_emb(clip_fea.float()).float(), context], dim=1)
            context_img_len += clip_fea.shape[-2]
        if extras.get("clip_fea_ref") is not None:
            context = torch.cat([net.img_emb_refimage(extras["clip_fea_ref"].float()).float(), context], dim=1)
            context_img_len += extras["clip_fea_ref"].shape[-2]
        if extras.get("audio_embed") is not None:
            audio_emb = music_embedding(net, extras["audio_embed"], grid_sizes[0])
    elif clip_fea is not None and kind != "humo":
        if net.img_emb is not None:
            context = torch.cat([net.img_emb(clip_fea.float()).float(), context], dim=1)
            context_img_len = clip_fea.shape[-2]
        elif kind != "vace":
            context_img_len = clip_fea.shape[-2]
    humo_audio = None
    if kind == "humo" and extras.get("audio_embed") is not None:
        audio_embed = extras["audio_embed"].float()
        if reference_latent is not None:
            zeros = audio_embed.new_zeros((audio_embed.shape[0], reference_latent.shape[-3]) + tuple(audio_embed.shape[2:]))
            audio_embed = torch.cat([audio_embed, zeros], dim=1)
        humo_audio = net.audio_proj(audio_embed).float()  # [B, F', 16, 1536]: кадр -> 16 аудио-токенов
        humo_frames = humo_audio.shape[1]

    vace = None
    if kind == "vace":
        vctx = extras.get("vace_context")
        if vctx is None:
            raise ValueError("VACE модель требует vace_context (WanVaceToVideo)")
        shape = list(vctx.shape)
        packed = vctx.float().movedim(0, 1).reshape([-1] + shape[2:])
        packed = comfy.ldm.common_dit.pad_to_patch_size(packed, net.patch_size)
        c_tokens = net.vace_patch_embedding(packed).flatten(2).transpose(1, 2)
        if c_tokens.shape[1] != seq_len:
            raise ValueError(f"VACE context {c_tokens.shape[1]} токенов != видео {seq_len}")
        strength = extras.get("vace_strength") or [1.0] * shape[1]
        vace = dict(c=list(c_tokens.split(shape[0], dim=0)), strength=[float(s) for s in strength],
                    context=context if context_img_len is None else context[:, context_img_len:])
        del packed, c_tokens

    total = tokens.shape[1]
    if freqs.shape[1] != total:
        raise ValueError(f"RoPE длина {freqs.shape[1]} != токенов {total}")
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    last_a, last_b = shard_bounds(total, world - 1, world)
    state.update(total=total, enabled=world > 1 and last_a < last_b, kv_collectives=0, kv_gathered_bytes=0,
                 out_collectives=0, out_exchanged_bytes=0, scale_collectives=0, head_gathers=0)
    if world > 1 and not state["enabled"] and not state.get("warned_empty"):
        import warnings
        warnings.warn("Wan sequence split создаёт пустой rank; этот forward считает всю последовательность "
                      "на каждом GPU (FSDP-only, duplicated compute)")
        state["warned_empty"] = True
    a, b = shard_bounds(total, rank, world) if state["enabled"] else (0, total)
    index = TokenIndex(a, b, total, e0.shape[1], tokens.device)
    local = tokens[:, a:b].contiguous()
    freqs_local = freqs[:, a:b]
    del tokens
    if vace is not None:
        vace["c"] = [cj[:, a:b].contiguous() for cj in vace["c"]]
        vace["x"] = local
    uni3c = prepare_uni3c(_powershard_uni3c_runtime, _powershard_uni3c, uni3c_render, x_input, e, options, total,
                          ref_len, (a, b), state)
    hooks = {}
    talk = prepare_multitalk(_powershard_multitalk_runtime, _powershard_multitalk, multitalk_audio, multitalk_masks,
                             state, total, grid_sizes[1] * grid_sizes[2], (a, b), ref_len)
    if talk is not None:
        hooks["multitalk"] = talk
    if humo_audio is not None:
        if total % humo_frames:
            raise ValueError(f"HuMo: {total} токенов не делятся на {humo_frames} аудио-кадров")
        hooks["humo"] = (humo_audio, frame_segments(a, b, total // humo_frames))
    face_rows = (a, b, total)
    inject_rows = None
    if audio_emb is not None:  # S2V audio_injector / WanDancer music_injector: строки [0, seq_len)
        frames = audio_emb.shape[1]
        if seq_len % frames:
            raise ValueError(f"{kind}: {seq_len} видео-токенов не делятся на {frames} аудио-кадров")
        hi = max(a, min(b, seq_len))
        inject_rows = (hi - a, frame_segments(a, hi, seq_len // frames))
    injector = net.music_injector if kind == "wandancer" else getattr(net, "audio_injector", None)
    inject_scale = float(extras.get("audio_inject_scale", 1.0)) if kind == "wandancer" else 1.0

    cache = getattr(net, "_ps_block_cache", None)
    spec = block_cache_spec(_powershard_block_cache, options, cache, len(net.blocks))
    if spec is not None and hasattr(net.blocks[0], "set_modules_to_forward_prefetch"):
        net.blocks[0].set_modules_to_forward_prefetch([])
    first_residual = start = None
    for i, block in enumerate(net.blocks):
        if spec is not None and i == 1:
            first_residual = [local - start]
            skip, entry = cache.decide(spec, first_residual, [1.0])
            if skip:
                local = local + cache.skip(entry)[0]
                state["blocks_skipped"] = True
                break
            start = local
        elif spec is not None and i == 0:
            start = local
        local = block(local, e0, freqs_local, context, context_img_len, index, hooks or None)
        if uni3c is not None:
            local = uni3c.step(i, local)
        if vace is not None:
            mapped = net.vace_layers_mapping.get(i)
            if mapped is not None:
                for j in range(len(vace["c"])):
                    skip, vace["c"][j] = net.vace_blocks[mapped](vace["c"][j], vace["x"], e0, freqs_local, vace["context"], index)
                    local = local + skip * vace["strength"][j]
        if motion_vec is not None and i % 5 == 0:
            local = local + net.face_adapter.fuser_blocks[i // 5](local, motion_vec, face_rows).float()
        if inject_rows is not None:
            local = s2v_inject(injector, local, i, audio_emb, audio_emb_global, inject_rows, inject_scale)
    if spec is not None and not state["blocks_skipped"] and first_residual is not None:
        cache.store(spec, first_residual, [local - start])
    out = (net.head_global if global_fps else net.head)(local, e, index)
    if state["enabled"]:
        out = gather_tokens(out, total)
        state["head_gathers"] += 1
    if ref_len:
        out = out[:, ref_len:]
    out = net.unpatchify(out[:, :math.prod(grid_sizes)], grid_sizes)
    if ref_frames:
        out = out[:, :, ref_frames:]
    return out[:, :, :t, :h, :w]


def s2v_inject(injector, x, block_id, audio_emb, audio_emb_global, rows, scale=1.0):
    """AudioInjector_WAN для локальных видео-строк. Вызов units одинаков на всех rank."""
    idx = injector.injected_block_id.get(block_id)
    if idx is None:
        return x
    count, segments = rows
    video = x[:, :count]
    if injector.enable_adain and injector.adain_mode == "attn_norm":
        hidden = injector.injector_adain_layers[idx](video, audio_emb_global[:, :, 0], segments)
    else:
        hidden = injector.injector_pre_norm_feat[idx](video)
    residual = injector.injector[idx](hidden, audio_emb, frame_index=segments)
    if count:
        x = x.clone()
        x[:, :count] = x[:, :count] + residual.float() * scale
    return x


# --------------------------------------------------------------- Uni3C (ControlNet)
class Uni3CStep:
    def __init__(self, root, hidden, temb, freqs, strength, repeat, layers):
        self.root, self.hidden, self.temb, self.freqs = root, hidden, temb, freqs
        self.strength, self.repeat, self.layers = strength, repeat, layers

    def step(self, block_index, local):
        if block_index >= self.layers:
            return local
        self.hidden, residual = self.root("block", block_index, self.hidden, self.temb, self.freqs)
        residual = residual.float() * self.strength
        if self.repeat > 1:
            residual = residual.repeat(self.repeat, 1, 1)
        return local + residual


def prepare_uni3c(holder, spec, render, x_input, e, options, total, ref_len, rows, state):
    """Uni3C ControlNet (WanUni3CCnetPatch): своя FSDP root, те же строки токенов."""
    if spec is None:
        return None
    if holder is None:
        raise RuntimeError("Uni3C ControlNet не загружен в worker")
    sigmas = options.get("sigmas")
    if sigmas is not None:
        sigma = float(sigmas.reshape(-1)[0])
        if sigma > spec["sigma_start"] or sigma < spec["sigma_end"]:
            return None
    if ref_len:
        raise ValueError("Uni3C + reference_latent (ref_conv) не поддерживается: токены сдвинуты")
    root = holder.root
    num_conds = len(options.get("cond_or_uncond", [0]))
    samples = x_input.shape[0]
    if num_conds > 0 and samples % num_conds == 0:
        samples //= num_conds
    hidden = x_input[:samples, :20].float()
    if hidden.shape[1] < 20:
        pad = list(hidden.shape)
        pad[1] = 20 - hidden.shape[1]
        hidden = torch.cat([hidden, hidden.new_zeros(pad)], dim=1)
    import comfy.ldm.common_dit
    render = comfy.ldm.common_dit.pad_to_patch_size(render.float().to(hidden.device), (1, 2, 2))
    if render.shape[2:] != hidden.shape[2:]:
        raise ValueError(f"Uni3C render latent {list(render.shape[2:])} != латент модели {list(hidden.shape[2:])}")
    if render.shape[0] != hidden.shape[0]:
        render = render[:1].expand(hidden.shape[0], -1, -1, -1, -1)
    control_hidden, freqs = root("input", torch.cat([hidden, render], dim=1))
    if control_hidden.shape[1] != total:
        raise ValueError(f"Uni3C токенов {control_hidden.shape[1]} != {total}")
    a, b = rows
    holder.state.update(total=total, enabled=state["enabled"], kv_collectives=0, kv_gathered_bytes=0,
                        out_collectives=0, out_exchanged_bytes=0, scale_collectives=0)
    temb = e[:samples]
    if temb.ndim == 3:
        temb = temb[:, 0]
    repeat = x_input.shape[0] // samples
    return Uni3CStep(root, control_hidden[:, a:b].float().contiguous(), temb.float(), freqs.float()[:, a:b],
                     float(spec["strength"]), repeat, root.network.num_layers)


# ------------------------------------------------------------ accelerators
def batch_total(options, batch):
    rows = options.get("_ps_rows")
    return rows[2] if rows else batch


def batch_rows(rows, options):
    """Строки полного batch -> строки текущей части (WanEntrypoint batch_chunk)."""
    part = options.get("_ps_rows")
    if not part:
        return rows
    a, b, _ = part
    return [r - a for r in rows if a <= r < b]


def block_cache_spec(spec, options, cache, blocks):
    if not spec or not spec.get("active") or cache is None or blocks < 2:
        return None
    part = options.get("_ps_rows")
    return dict(spec, key=spec["key"] + (f"|rows{part[0]}-{part[1]}" if part else ""))


class riflex_context:
    """RIFLEx на время вычисления RoPE основной последовательности (rope_embedder экземпляра)."""

    def __init__(self, net, spec, latent_frames):
        self.net, self.spec, self.frames = net, spec, latent_frames
        self.original = None

    def __enter__(self):
        if not self.spec:
            return self
        from .accel import riflex_embedder, riflex_frequency_index
        embedder = self.net.rope_embedder
        head_dim = self.net.dim // self.net.num_heads
        k = int(self.spec.get("k") or 0) or riflex_frequency_index(head_dim, int(self.spec.get("train_frames", 21)))
        temporal = head_dim - 4 * (head_dim // 6)
        if not 1 <= k <= temporal // 2:
            raise ValueError(f"RIFLEx k={k} вне 1..{temporal // 2}")
        self.original, forward = riflex_embedder(embedder, k, int(round(self.frames)))
        embedder.forward = forward
        return self

    def __exit__(self, *exc):
        if self.original is not None:
            del self.net.rope_embedder.forward  # снять instance attribute: снова метод класса
        return False
