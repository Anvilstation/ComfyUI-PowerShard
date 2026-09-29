"""Реальные native ComfyUI классы уменьшенной размерности; это НЕ CUDA/FSDP тест."""
import copy
import json
from pathlib import Path
import pytest
import torch
from powershard.config import DistributedConfig
from powershard.qwen import QwenOperations, QwenEntrypoint, install_qwen_compute, infer_qwen_config
from powershard.attention_policy import AttentionDispatcher
from powershard.operations import Linear, Int8Linear, regular_hadamard, dequantize_rows


@pytest.fixture
def qwen_factory(h3_factory,monkeypatch):
    from comfy.text_encoders.minimax import MiniMaxH3TEModel
    import comfy.text_encoders.qwen3vl as native
    monkeypatch.setitem(native.QWEN3VL_VISION,"qwen3vl_32b",dict(hidden_size=64,intermediate_size=128,
        depth=2,deepstack_visual_indexes=[0,1]))
    def make():
        torch.manual_seed(2048)
        encoder=MiniMaxH3TEModel(device="cpu",dtype=torch.float32,model_options={"custom_operations":QwenOperations,
            "qwen3vl_32b_model_config":dict(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                num_attention_heads=2,num_key_value_heads=1)})
        encoder.eval().requires_grad_(False)
        for n,p in encoder.named_parameters():
            p.copy_(torch.ones_like(p) if "norm" in n and n.endswith("weight") else torch.randn_like(p)*.02)
        for m in encoder.modules():
            if isinstance(m,Linear):m._ps_safe=True;m._ps_fp32=True
        # Native special pad ID is outside the deliberately reduced test vocab.
        encoder.qwen3vl_32b.special_tokens={"pad":0}
        return encoder
    return make


@pytest.mark.parametrize("image",[False,True,"video"])
@pytest.mark.parametrize("quant",[False,True])
@pytest.mark.parametrize("profile",["custom","ram_min"])
def test_native_qwen_encoder_conditioning(qwen_factory,image,quant,profile):
    native=qwen_factory();encoder=copy.deepcopy(native)
    if quant:
        net=encoder.qwen3vl_32b.transformer
        for name,m in list(net.named_modules()):
            if not isinstance(m,Linear) or not name.startswith("model.layers."):continue
            group=16
            w=m.weight.detach().float()
            rotated=(w.reshape(w.shape[0],-1,group)@regular_hadamard(group,"cpu")).reshape_as(w)
            scale=rotated.abs().amax(-1,keepdim=True)/127
            q=Int8Linear(m,dict(convrot=True,group_size=group),8)
            q.weight=torch.nn.Parameter((rotated/scale).round().to(torch.int8),requires_grad=False)
            q.weight_scale=torch.nn.Parameter(scale,requires_grad=False)
            parent,_,leaf=name.rpartition(".");setattr(net.get_submodule(parent),leaf,q)
            # Reference uses the SAME dequantized weights. Quantization error vs
            # original weights is a different comparison from compute correctness.
            native.qwen3vl_32b.transformer.get_submodule(name).weight.copy_(
                dequantize_rows(q.weight,q.weight_scale,True,group,dtype=torch.float32))
    else:
        encoder.half() # only dense test reference; never applied to quantized runtime
    cfg=DistributedConfig(gpu_ids=("0",),attention_backend="sdpa",memory_profile=profile,dequant_rows=8,workspace_mib=1)
    dispatch=AttentionDispatcher(cfg)
    tracker=install_qwen_compute(encoder.qwen3vl_32b.transformer,dispatch)
    if cfg.min_vram:
        from powershard.memory_policy import apply_linear_policy,apply_workspace_policy
        apply_linear_policy(encoder.qwen3vl_32b.transformer,cfg)
        for layer in encoder.qwen3vl_32b.transformer.model.layers:
            layer.mlp._ps_memory_context=apply_workspace_policy(dict(mlp_budget_bytes=16384),cfg)
    root=QwenEntrypoint(encoder);root._ps_device=torch.device("cpu")
    tokens=[(10,1.),(11,1.),(12,1.)]
    if image:
        entry={"type":"image","data":torch.rand(2 if image=="video" else 1,64,64,3),"original_type":"image"}
        if image=="video":entry["minimax_video_block"]=True
        tokens.insert(1,(entry,1.))
    tokens={"qwen3vl_32b":[tokens]}
    native.set_clip_options({"execution_device":torch.device("cpu")})
    with torch.no_grad():
        reference=native.encode_token_weights(tokens)
        tracker.begin(torch.device("cpu"))
        actual=root("encode",(tokens,),{})
        tracker.finish(actual[0])
    assert actual[0].shape==reference[0].shape
    assert actual[0].dtype==torch.float32 and actual[1] is None
    torch.testing.assert_close(actual[0],reference[0],rtol=.025,atol=.006)
    torch.testing.assert_close(actual[2]["minimax_token_tags"],reference[2]["minimax_token_tags"])
    assert dispatch.report()["call_counts"]
    if cfg.min_vram:
        assert any(key.endswith(':sdpa_tiled') for key in dispatch.report()['call_counts'])
    if quant:assert any(p.dtype==torch.int8 for p in root.parameters())


def test_real_header_native_geometry(h3_factory):
    root=Path(__file__).resolve().parents[1]
    header=json.loads((root/"models/qwen3vl_32b_minimax_h3_bf16.safetensors.header.json").read_text())
    conf=infer_qwen_config(header)
    from comfy.text_encoders.minimax import MiniMaxH3TEModel
    import inspect
    from comfy.text_encoders.llama import Qwen3VL_32BConfig
    with torch.device("meta"):
        encoder=MiniMaxH3TEModel(device="meta",dtype=torch.float16,model_options={"custom_operations":QwenOperations,
            "qwen3vl_32b_model_config":{k:v for k,v in conf.items() if k in inspect.signature(Qwen3VL_32BConfig).parameters}})
    tensors=encoder.qwen3vl_32b.transformer.state_dict()
    assert set(tensors)==set(header)-{"__metadata__"}
    assert all(list(v.shape)==header[k]["shape"] for k,v in tensors.items())
    assert encoder.qwen3vl_32b.layer=="last" and encoder.qwen3vl_32b.transformer.model.norm is None


def test_cache_bound_clone_and_hash():
    from powershard.conditioning_cache import ConditioningCache, content_hash
    cache=ConditioningCache(16)
    a=torch.tensor([1.,2.]);key=content_hash({"tokens":a,"precision":"fp16"})
    cache.put(key,(a,None,{"tags":torch.tensor([1,0],dtype=torch.int32)}))
    v=cache.get(key);v[0].fill_(99)
    assert cache.get(key)[0][0]==1
    assert key!=content_hash({"tokens":a+1,"precision":"fp16"})
    assert key!=content_hash({"tokens":a,"precision":"int8_fp16"})
    cache.put("second",torch.zeros(4));assert cache.get(key) is None
    assert cache.bytes<=cache.limit
    cache.clear();assert not cache.bytes


def test_native_clip_wrapper_cache_and_clone(qwen_factory,tmp_path):
    from powershard.qwen import QwenConfig
    from powershard.qwen_adapter import DistributedQwenCLIP,QwenProxy,QwenPatcher
    from powershard.runtime import Session
    from powershard.patch_config import H3PatchConfig
    native=qwen_factory()
    dispatch=AttentionDispatcher(DistributedConfig(attention_backend="sdpa"))
    install_qwen_compute(native.qwen3vl_32b.transformer,dispatch)
    root=QwenEntrypoint(native);root._ps_device=torch.device("cpu")
    path=tmp_path/"identity-only";path.write_bytes(b"test")
    class LocalSession(Session):
        calls=0
        def call(self,command,args,kwargs,cancel=None):
            self.calls+=1
            with torch.no_grad():return root(command,args,kwargs)
    options=QwenConfig(cache_mib=1)
    session=LocalSession(str(path),DistributedConfig(),str(tmp_path),patch=H3PatchConfig(),role="qwen",role_options=options.to_dict())
    proxy=QwenProxy(session,options,"tiny-native-tokenizer")
    clip=DistributedQwenCLIP(no_init=True)
    clip.cond_stage_model=proxy;clip.patcher=QwenPatcher(proxy,torch.device("cpu"),torch.device("cpu"),size=1)
    clip.tokenizer=None;clip.layer_idx=None;clip.tokenizer_options={};clip.use_clip_schedule=False;clip.apply_hooks_to_conds=None
    tokens={"qwen3vl_32b":[[(10,1.),(11,1.)]]}
    first=clip.encode_from_tokens_scheduled(tokens)
    second=clip.encode_from_tokens_scheduled(tokens)
    assert session.calls==1 and proxy.last_encoding["cache_hit"]
    assert "minimax_token_tags" in first[0][1] and first[0][1]["pooled_output"] is None
    torch.testing.assert_close(first[0][0],second[0][0])
    clone=clip.clone();clone.patcher.model_options["alien_patch"]={"x":1}
    assert not clip.patcher.model_options.get("alien_patch")
    assert clone.cond_stage_model is not proxy and clone.cond_stage_model.cache is not proxy.cache
    with pytest.raises(ValueError,match="model_options"):clone.encode_from_tokens_scheduled(tokens)
    first[0][0].fill_(1234)
    assert not torch.equal(first[0][0],clip.encode_from_tokens_scheduled(tokens)[0][0])
    clip.clear_cache();clip.encode_from_tokens_scheduled(tokens);assert session.calls==2
    changed={"qwen3vl_32b":[[(10,1.),(12,1.)]]}
    clip.encode_from_tokens_scheduled(changed);assert session.calls==3
    with pytest.raises(ValueError,match="Remote CLIP"):clip.get_sd()


def test_qwen_memory_plan_uses_per_layer_groups():
    from powershard.qwen import qwen_memory_plan
    root=Path(__file__).resolve().parents[1]
    header=json.loads((root/"models/qwen3vl_32b_minimax_h3_int8_convrot.safetensors.header.json").read_text())
    plan=qwen_memory_plan(header,3)
    assert len([g for g in plan["groups"] if g.startswith("model.layers.")])==50
    assert plan["largest_group_bytes_upper_bound"]<plan["converted_storage_bytes"]*.1
    offload=qwen_memory_plan(header,3,cpu_offload=True)
    assert plan["host_gpu_parameter_budget_bytes"]-offload["host_gpu_parameter_budget_bytes"]==plan["shard_bytes_lower_bound"]


@pytest.mark.parametrize("world",[1,2,3,6])
@pytest.mark.parametrize("mrope",[False,True])
def test_sequence_math_with_injected_collective_payloads(qwen_factory,monkeypatch,world,mrope):
    """Проверяет РАЗРЕЗЫ/causal/GQA/RoPE, а не реальный transport/NCCL."""
    from comfy.text_encoders.llama import precompute_freqs_cis,apply_rope
    from powershard.config import shard_bounds
    import powershard.attention as attention
    import torch.distributed as dist
    ref=qwen_factory().qwen3vl_32b.transformer
    seq=qwen_factory().qwen3vl_32b.transformer
    config=DistributedConfig(attention_backend="sdpa")
    install_qwen_compute(ref,AttentionDispatcher(config))
    install_qwen_compute(seq,AttentionDispatcher(config),sequence=True)
    layer,local=ref.model.layers[0],seq.model.layers[0]
    B,L,H=2,11,32
    torch.manual_seed(50);h=torch.randn(B,L,H)
    positions=torch.arange(L).unsqueeze(0).repeat(3 if mrope else B,1)
    if mrope:positions[1]*=2;positions[2]*=3
    freqs=precompute_freqs_cis(layer.self_attn.head_dim,positions,10000.,
                              rope_dims=[24,20,20] if mrope else None,interleaved_mrope=mrope)
    mask=torch.full((1,1,L,L),float('-inf')).triu(1)
    with torch.no_grad():
        expected,_=layer(h.clone(),mask,freqs)
        normalized=layer.input_layernorm(h);a=layer.self_attn
        q=a.q_norm(a.q_proj(normalized).view(B,L,a.num_heads,a.head_dim).transpose(1,2))
        k=a.k_norm(a.k_proj(normalized).view(B,L,a.num_kv_heads,a.head_dim).transpose(1,2))
        q,k=apply_rope(q,k,freqs)
        v=a.v_proj(normalized).view(B,L,a.num_kv_heads,a.head_dim)
        full_kv=torch.stack((k.transpose(1,2),v),dim=2).transpose(0,1).contiguous()
        for rank in range(world):
            start,end=shard_bounds(L,rank,world);seen=[]
            monkeypatch.setattr(dist,"get_world_size",lambda:world)
            monkeypatch.setattr(dist,"get_rank",lambda:rank)
            def fake_gather(x,total):
                assert total==L;seen.append(tuple(x.shape))
                if x.ndim==5:
                    torch.testing.assert_close(x,full_kv[start:end],rtol=.003,atol=.002)
                    return full_kv
                torch.testing.assert_close(x,expected[:,start:end].transpose(0,1),rtol=.01,atol=.002)
                return expected.transpose(0,1)
            monkeypatch.setattr(attention,"gather_rows",fake_gather)
            actual,_=local(h.clone(),mask,freqs)
            torch.testing.assert_close(actual,expected,rtol=.01,atol=.002)
            assert len(seen)==(2 if world>1 else 0)
