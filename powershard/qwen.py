"""Родной H3 Qwen3-VL: tokenizer/hidden states/vision принадлежат ComfyUI.

PowerShard меняет размещение и compute operations, не семантику conditioning.
"""
from dataclasses import dataclass, asdict
import math
import types
import warnings
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from .operations import Operations, Linear, Int8Linear, install_int8
from .fp16_safe import FiniteTracker, power2_scale
from .checkpoint import Checkpoint
from .config import shard_bounds


@dataclass(frozen=True)
class QwenConfig:
    idle_policy: str = "release"
    cache_mib: int = 256
    mlp_chunk_mode: str = "auto"
    mlp_chunk_tokens: int = 4096

    def __post_init__(self):
        if self.idle_policy not in ("release","cpu_shards","keep") or self.cache_mib < 0:
            raise ValueError("Qwen: idle_policy=release/cpu_shards/keep, cache_mib >= 0")
        if self.mlp_chunk_mode not in ("auto","manual","off") or self.mlp_chunk_tokens<1:
            raise ValueError("Qwen MLP: auto/manual/off, chunk_tokens >= 1")

    def to_dict(self):
        return asdict(self)


def infer_qwen_config(tensors):
    """Фактическая глубина/размеры из checkpoint, остальная архитектура native."""
    def shape(k):
        try:return tensors[k]["shape"]
        except KeyError as e:raise ValueError(f"Не H3 Qwen3-VL checkpoint: отсутствует {k}") from e
    layers={int(k.split('.')[2]) for k in tensors if k.startswith('model.layers.')}
    if not layers or layers!=set(range(len(layers))):
        raise ValueError("Непоследовательные decoder layers Qwen")
    dim=shape('model.layers.0.self_attn.q_norm.weight')[0]
    vocab,hidden=shape('model.embed_tokens.weight')
    if any(k.startswith(('lm_head.','model.norm.')) for k in tensors):
        raise ValueError("H3 conditioning требует truncated unnormalized encoder без lm_head/final norm")
    return dict(vocab_size=vocab,hidden_size=hidden,num_hidden_layers=len(layers),
                intermediate_size=shape('model.layers.0.mlp.gate_proj.weight')[0],
                num_attention_heads=shape('model.layers.0.self_attn.q_proj.weight')[0]//dim,
                num_key_value_heads=shape('model.layers.0.self_attn.k_proj.weight')[0]//dim,
                head_dim=dim,final_norm=False,lm_head=False)


class Embedding(nn.Embedding):
    def reset_parameters(self):pass
    def forward(self, x, out_dtype=None):
        return F.embedding(x,self.weight,self.padding_idx,self.max_norm,self.norm_type,
                           self.scale_grad_by_freq,self.sparse).to(out_dtype or self.weight.dtype)


class LayerNorm(nn.LayerNorm):
    def reset_parameters(self):pass
    def forward(self,x):
        return F.layer_norm(x.float(),self.normalized_shape,None if self.weight is None else self.weight.float(),
                            None if self.bias is None else self.bias.float(),self.eps)


class Conv3d(nn.Conv3d):
    def reset_parameters(self):pass
    def forward(self,x):
        # Небольшая vision patch projection — FP32 safety island.
        return F.conv3d(x.float(),self.weight.float(),None if self.bias is None else self.bias.float(),
                        self.stride,self.padding,self.dilation,self.groups)


class QwenOperations(Operations):
    Embedding=Embedding
    LayerNorm=LayerNorm
    Conv3d=Conv3d


def qwen_attention(self,hidden_states,attention_mask=None,freqs_cis=None,optimized_attention=None,
                   past_key_value=None,sliding_window=None):
    from comfy.text_encoders.llama import apply_rope
    from .attention_contract import AttentionOptions
    from .attention import gather_rows
    if past_key_value is not None or sliding_window is not None:
        raise ValueError("H3 CLIP encoding не поддерживает autoregressive KV cache/sliding window")
    batch,length,_=hidden_states.shape
    q=self.q_proj(hidden_states).view(batch,length,self.num_heads,self.head_dim).transpose(1,2)
    k=self.k_proj(hidden_states).view(batch,length,self.num_kv_heads,self.head_dim).transpose(1,2)
    v=self.v_proj(hidden_states).view(batch,length,self.num_kv_heads,self.head_dim)
    q,k=self.q_norm(q),self.k_norm(k)
    q,k=apply_rope(q,k,freqs_cis)
    q,k=q.transpose(1,2),k.transpose(1,2)
    seq=getattr(self,"_ps_qwen_sequence",None)
    if seq and seq["enabled"]:
        kv=torch.stack((k,v),dim=2).transpose(0,1).contiguous()
        kv=gather_rows(kv,seq["total"]).transpose(0,1)
        k,v=kv.unbind(2)
        seq["kv_all_gathers"]+=1
    scale=self.head_dim**-.5
    sq,sk=self._ps_qk_scales
    sv=power2_scale(v.float().abs().amax()/16384.) if v.numel() else 1.
    q,k,v=(q.float()/sq).half(),(k.float()/sk).half(),(v.float()/sv).half()
    out=self._ps_dispatch(q,k,v,AttentionOptions(softmax_scale=scale*sq*sk,mask=attention_mask),group="qwen_text",compute_fp16=True)
    self._ps_shapes=dict(q=list(q.shape),k=list(k.shape),v=list(v.shape),mask=list(attention_mask.shape) if attention_mask is not None else None)
    return self.o_proj((out.float()*sv).reshape(batch,length,self.inner_size)),None


def qwen_mlp(self,x):
    from .memory_policy import mlp_plan
    from .operations import prepared_linears
    shape=x.shape;rows=x.reshape(-1,shape[-1]);options=self._ps_qwen_options
    budget=self._ps_memory_context.get("mlp_budget_bytes",256*2**20)
    mode=self._ps_memory_context.get("mlp_mode_override",options.mlp_chunk_mode)
    plan=mlp_plan(len(rows),shape[-1],self.down_proj.in_features,mode,options.mlp_chunk_tokens,budget)
    plan["requested_mode"]=options.mlp_chunk_mode
    out=torch.empty_like(rows,dtype=torch.float32)
    allowance=max(0,budget-plan["estimated_chunk_workspace_bytes"]-out.numel()*4)
    with prepared_linears((self.gate_proj,self.up_proj,self.down_proj),allowance,
            enabled=plan["chunks"]>1 and self._ps_memory_context.get("allow_prepared_weights",True)) as preparation:
        plan.update(preparation)
        for a in range(0,len(rows),plan["effective_tokens"]):
            b=min(a+plan["effective_tokens"],len(rows))
            gate,up=self.gate_proj(rows[a:b]),self.up_proj(rows[a:b])
            out[a:b]=self.down_proj(F.silu(gate.float())*up.float())
            del gate,up
    self._ps_mlp_report=plan
    return out.reshape(shape)


def install_qwen_compute(net, dispatcher, checkpoint=None, sequence=False, options=None):
    """Родные RoPE/DeepStack/layer selection сохранены; Qwen-specific FP32 stream."""
    tracker=FiniteTracker(False)
    for name,m in net.named_modules():
        if isinstance(m,(Linear,Int8Linear)):
            m._ps_safe=True;m._ps_fp32=False
            m._ps_tracker=tracker;m._ps_finite_slot=tracker.register(name)
    from safetensors import safe_open
    reader=safe_open(str(checkpoint.path),framework="pt",device="cpu") if checkpoint else None
    for i,layer in enumerate(net.model.layers):
        layer.mlp._ps_qwen_options=options or QwenConfig()
        layer.mlp._ps_memory_context={}
        layer.mlp.forward=types.MethodType(qwen_mlp,layer.mlp)
        attn=layer.self_attn
        attn._ps_dispatch=dispatcher
        scales=[]
        for norm in ("q_norm","k_norm"):
            w=reader.get_tensor(f"model.layers.{i}.self_attn.{norm}.weight") if reader else getattr(attn,norm).weight.detach()
            bound=2*math.sqrt(attn.head_dim)*float(w.float().abs().max())*1.01
            scales.append(2.**max(0,math.ceil(math.log2(max(1.,bound/128.)))))
        attn._ps_qk_scales=scales
        attn.forward=types.MethodType(qwen_attention,attn)
        if sequence:
            original=layer.forward
            state={"enabled":False,"total":0,"kv_all_gathers":0,"hidden_all_gathers":0}
            attn._ps_qwen_sequence=state
            def run(self,x,attention_mask=None,freqs_cis=None,optimized_attention=None,past_key_value=None,
                    _original=original,_state=state):
                from .attention import gather_rows
                total=x.shape[1];world=dist.get_world_size();rank=dist.get_rank()
                last=shard_bounds(total,world-1,world)
                _state.update(total=total,enabled=world>1 and last[1]>last[0])
                _state["fallback_reason"] = None if _state["enabled"] else ("world_size=1" if world==1 else "ceil token split has empty last rank")
                if not _state["enabled"]:
                    return _original(x,attention_mask,freqs_cis,optimized_attention,past_key_value)
                a,b=shard_bounds(total,rank,world)
                _state["local_range"]=[a,b]
                # Native precompute: [B,1,L,D] (text) or [1,L,D]
                # (interleaved multimodal RoPE). Sequence axis is explicitly -2.
                if isinstance(freqs_cis,tuple):
                    if any(f.ndim not in (3,4) or f.shape[-2]!=total for f in freqs_cis):
                        raise ValueError("Qwen sequence: неизвестный native RoPE tensor layout")
                    local_freqs=tuple(f[...,a:b,:] for f in freqs_cis)
                else:
                    raise ValueError("Qwen sequence: нужен native RoPE layout (cos,sin,-sin)")
                mask=None if attention_mask is None else attention_mask[...,a:b,:]
                y,kv=_original(x[:,a:b].clone(),mask,local_freqs,optimized_attention,past_key_value)
                full=gather_rows(y.transpose(0,1).contiguous(),total).transpose(0,1)
                _state["hidden_all_gathers"]+=1
                return full,kv
            layer.forward=types.MethodType(run,layer)
    # Vision keeps native rotary/batch boundaries. Dispatch is injected into each
    # native attention instance; no global optimized_attention replacement.
    for block in net.visual.blocks:
        attn=block.attn
        original=attn.forward
        def vision(self,x,cu_seqlens,position_embeddings,optimized_attention=None,_original=original):
            from .attention_contract import AttentionOptions
            def attend(q,k,v,heads,mask=None,skip_reshape=False,**kw):
                # Native vision call contract B,H,L,D; QK lacks RMS bound, so
                # exact FP32 SDPA/math safety path remains explicit in telemetry.
                return dispatcher(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                    AttentionOptions(mask=mask),group="qwen_vision",compute_fp16=False).reshape(q.shape[0],q.shape[2],-1)
            return _original(x,cu_seqlens,position_embeddings,optimized_attention=attend)
        attn.forward=types.MethodType(vision,attn)
    return tracker


def qwen_memory_plan(tensors,world,prefetch=0,cpu_offload=False):
    from .checkpoint import BYTES
    groups={};stored=0
    for key,info in tensors.items():
        if key=="__metadata__":continue
        n=math.prod(info["shape"]);stored+=n*BYTES[info["dtype"]]
        if key.endswith(".comfy_quant"):continue
        size=n*(1 if info["dtype"]=="I8" else 4 if info["dtype"]=="F32" else 2)
        parts=key.split(".")
        if key.startswith("model.layers."):group=".".join(parts[:3])
        elif key.startswith("visual.blocks."):group=".".join(parts[:3])
        elif key.startswith("visual.deepstack_merger_list."):group=".".join(parts[:3])
        elif key.startswith(("visual.patch_embed.","visual.merger.","model.embed_tokens.")):group=".".join(parts[:2])
        elif key.startswith("visual."):group="visual_root"
        else:group="root"
        groups[group]=groups.get(group,0)+size
    total=sum(groups.values());largest=max(groups.values(),default=0)
    bound=groups.get("root",0)+groups.get("visual_root",0)+sum(sorted(groups.values(),reverse=True)[:1+prefetch])
    return dict(checkpoint_tensor_bytes=stored,converted_storage_bytes=total,world_size=world,
        shard_bytes_lower_bound=math.ceil(total/world),largest_group_bytes_upper_bound=largest,groups=groups,
        host_gpu_parameter_budget_bytes=(0 if cpu_offload else math.ceil(total/world))+bound,
        note_ru="Parameter estimate only: padding/activations/kernel/dequant/allocator/NCCL добавляются отдельно")


class QwenEntrypoint(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.network=encoder.qwen3vl_32b.transformer
        # Не регистрировать второй alias того же многогигабайтного module tree.
        object.__setattr__(self,"encoder",encoder)

    def forward(self,command,args,kwargs):
        if command!="encode":raise ValueError(f"Неизвестная операция Qwen: {command}")
        self.encoder.reset_clip_options()
        options=dict(kwargs.get("clip_options",{}))
        options["execution_device"]=self._ps_device
        self.encoder.set_clip_options(options)
        return self.encoder.encode_token_weights(args[0])
