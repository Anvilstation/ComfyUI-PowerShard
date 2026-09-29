import copy
import pytest
import torch
from powershard.fp16_safe import safe_linear, scaled_matmul, FiniteTracker, apply_fp16_safe
from powershard.operations import Linear, Int8Linear, dequantize_rows
from powershard.patch_config import H3PatchConfig


def test_old_condition_overflow_and_safe_fp32():
    old = Linear(24, 32, bias=False, dtype=torch.float16).requires_grad_(False)
    old.weight.fill_(1.)
    x = torch.full((7,24), 100000.)
    with pytest.raises(FloatingPointError):old(x)
    old._ps_safe = True
    old._ps_fp32 = True
    got = old(x)
    assert got.dtype == torch.float32 and got.shape == (7,32)
    torch.testing.assert_close(got, torch.full_like(got, 2400000.), rtol=0, atol=0)


@pytest.mark.parametrize("magnitude", [1., 1000., 1e6])
def test_scaled_projection_and_bias(magnitude):
    g = torch.Generator().manual_seed(42)
    x = torch.randn(13,64,generator=g)*magnitude
    w = torch.randn(17,64,generator=g).half()
    bias = torch.randn(17,generator=g)*100
    got = safe_linear(x,w,bias)
    expected = torch.nn.functional.linear(x,w.float(),bias)
    assert got.dtype==torch.float32 and torch.isfinite(got).all()
    error = (got-expected).square().mean().sqrt()/expected.square().mean().sqrt()
    assert error < .001
    torch.testing.assert_close(got,expected,atol=magnitude*.012,rtol=.003)


def test_attention_matmuls_large_values():
    from powershard.attention import exact_attention
    g=torch.Generator().manual_seed(9)
    q=torch.randn(7,11,8,generator=g);k=torch.randn(7,13,8,generator=g)
    v=torch.randn(7,13,8,generator=g)*1e6
    got=exact_attention(q,k,v,4,5,compute_fp16=True)
    ref=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    assert torch.isfinite(got).all()
    assert ((got-ref).square().mean()/ref.square().mean()).sqrt() < .002


def test_int8_convrot_scaled_storage():
    src=Linear(256,13,bias=False,device="meta",dtype=torch.float16)
    layer=Int8Linear(src,{"convrot":True,"group_size":256},rows=4).to_empty(device="cpu")
    g=torch.Generator().manual_seed(22)
    layer.weight.copy_(torch.randint(-20,20,layer.weight.shape,generator=g,dtype=torch.int8))
    layer.weight_scale.fill_(.01)
    layer._ps_safe=True;layer._ps_fp32=False
    x=torch.randn(7,256,generator=g)*1e6
    got=layer(x)
    w=dequantize_rows(layer.weight,layer.weight_scale,True,256,dtype=torch.float32)
    ref=torch.nn.functional.linear(x,w)
    assert torch.isfinite(got).all() and layer.weight.dtype==torch.int8
    assert ((got-ref).square().mean()/ref.square().mean()).sqrt() < .002
    assert not list(layer.buffers())
    assert sum(p.numel()*p.element_size() for p in layer.parameters())==13*256+13*4


def test_deferred_check_no_tensor_item_until_boundary(monkeypatch):
    tracker=FiniteTracker(debug=True);slot=tracker.register("mlp.fc2");tracker.begin("cpu")
    def forbidden(*args,**kwargs):raise AssertionError("synchronous tensor read inside compute")
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor,"item",forbidden)
        m.setattr(torch.Tensor,"__bool__",forbidden)
        out=safe_linear(torch.ones(3,4)*1e5,torch.ones(2,4).half())
        tracker.observe(slot,out)
        tracker.observe(slot,torch.tensor([float("inf")]))
    with pytest.raises(FloatingPointError,match="mlp.fc2"):tracker.finish(out)


def test_policy_serialization():
    p=H3PatchConfig(enabled=True,debug_finite=True)
    assert H3PatchConfig(**p.to_dict()).fingerprint()==p.fingerprint()
    assert not H3PatchConfig(enabled=False).active


def test_h3_residual_mlp_condition_and_idempotency(h3_factory):
    from powershard.fsdp_backend import Entrypoint
    net=h3_factory()
    native_forward=type(net.blocks[0]).forward
    old=copy.deepcopy(net)
    net.condition_proj.weight.fill_(1.);old.condition_proj.weight.fill_(1.)
    text=torch.full((1,7,24),1e5)
    with pytest.raises(FloatingPointError):old.preprocess_text_embeds(text)
    policy=H3PatchConfig(enabled=True,debug_finite=True,mlp_chunk_tokens=3)
    tracker=apply_fp16_safe(net,policy)
    assert apply_fp16_safe(net,policy) is tracker
    assert type(net.blocks[0]).forward is native_forward
    tracker.begin("cpu")
    context=Entrypoint(net)("preprocess_text",(text,),{})
    tracker.finish(context)
    assert context.shape==(1,7,32) and context.dtype==torch.float32
    block=net.token_refiner.blocks[0]
    tracker.begin("cpu")
    out=block(torch.full((7,32),1e6))
    tracker.finish(out)
    assert out.dtype==torch.float32 and out.abs().max()>65504
    mlp=net.blocks[0].mlp
    mlp.fc1.weight.fill_(10.);mlp.fc2.weight.fill_(10.)
    x=torch.ones(7,32)
    tracker.begin("cpu");got=mlp(x);tracker.finish(got)
    u=torch.nn.functional.linear(x,mlp.fc1.weight.float());gate,up=u.chunk(2,-1)
    ref=torch.nn.functional.linear(torch.nn.functional.silu(gate)*up,mlp.fc2.weight.float())
    assert got.abs().max()>65504
    torch.testing.assert_close(got,ref,atol=32,rtol=.003)
    with pytest.raises(FloatingPointError):old.preprocess_text_embeds(text)


@pytest.mark.parametrize('case',['text','masks','fl2va','ref2va','audio_scale'])
def test_full_h3_safe_matches_fp32(h3_factory,case):
    net=h3_factory();ref=copy.deepcopy(net).float();ref.dtype=torch.float32
    tracker=apply_fp16_safe(net,H3PatchConfig(enabled=True,debug_finite=True,mlp_chunk_tokens=3))
    g=torch.Generator().manual_seed(71)
    video=torch.randn(1,24,2,4,4,generator=g);audio=torch.randn(1,32,2,5,generator=g)
    text=torch.randn(1,7,24,generator=g)*1e5
    payload={};extra={}
    if case=='masks':extra={'denoise_mask':torch.rand(1,1,2,4,4,generator=g),'audio_denoise_mask':torch.rand(1,1,2,5,generator=g)}
    if case=='fl2va':payload={'keyframes':[{'resolved_frame_index':0,'latent':video[:,:,:1],'audio_latent':audio[:,:,:,:2]}],
        'cond_video_latents':[video[:,:,:1]],'cond_audio_latents':[audio[:,:,:,:2]],'text_token_tags':torch.tensor([[1,0,0,1,1,1,1]]),'seed':77}
    if case=='ref2va':payload={'refs':[{'kind':'image','latent_h':4,'latent_w':4,'latent':video[:,:,:1]}],'cond_video_latents':[video[:,:,:1]],'seed':79}
    if case=='audio_scale':payload={'audio_scale':4.}
    with torch.no_grad():
        tracker.begin('cpu');context=net.preprocess_text_embeds(text);tracker.finish(context)
        context_ref=ref.preprocess_text_embeds(text)
        torch.testing.assert_close(context,context_ref,atol=.03,rtol=.003)
        tracker.begin('cpu');actual=net([video,audio],torch.tensor([700.]),context,minimax_payload=payload,**extra);tracker.finish(actual)
        expected=ref([video,audio],torch.tensor([700.]),context_ref,minimax_payload=payload,**extra)
        for x,y in zip(actual,expected):
            assert x.dtype==torch.float32 and torch.isfinite(x).all()
            torch.testing.assert_close(x,y,atol=.003,rtol=.003)


def test_native_h3_int8_convrot256_safe_forward(h3_factory):
    from powershard.operations import install_int8,regular_hadamard
    net=h3_factory(hidden_size=256,ffn_hidden_size=512,text_dim=256)
    original=dict(net.named_modules())
    mapping={name:{'convrot':True,'group_size':256} for name,m in original.items()
             if isinstance(m,Linear) and m.in_features%256==0 and m.weight.dtype==torch.float16}
    install_int8(net,mapping,64)
    for name in mapping:
        layer=net.get_submodule(name).to_empty(device='cpu')
        old=original[name];w=old.weight.float()
        rotated=(w.reshape(w.shape[0],-1,256)@regular_hadamard(256,'cpu')).reshape_as(w)
        scale=rotated.abs().amax(-1,keepdim=True)/127
        layer.weight.copy_((rotated/scale).round().to(torch.int8));layer.weight_scale.copy_(scale)
        if old.bias is not None:layer.bias.copy_(old.bias)
    tracker=apply_fp16_safe(net,H3PatchConfig(enabled=True,debug_finite=True,mlp_chunk_tokens=3))
    with torch.no_grad():
        tracker.begin('cpu');ctx=net.preprocess_text_embeds(torch.full((1,7,256),1e5));tracker.finish(ctx)
        tracker.begin('cpu');out=net([torch.zeros(1,24,2,4,4),torch.zeros(1,32,2,5)],torch.tensor([700.]),ctx);tracker.finish(out)
    assert len(out)==2 and ctx.dtype==torch.float32
    assert all(net.get_submodule(name).weight.dtype==torch.int8 for name in mapping)


def test_safe_vector_linear():
    actual=safe_linear(torch.ones(4)*1e5,torch.ones(2,4).half(),torch.ones(2))
    torch.testing.assert_close(actual,torch.full((2,),400001.),atol=0,rtol=.001)


def test_quantized_native_fp32_island_is_preserved():
    from powershard.fp16_safe import install_safe_operations
    src=Linear(256,4,bias=False,device='meta',dtype=torch.float32)
    layer=Int8Linear(src,{'convrot':True,'group_size':256},rows=2).to_empty(device='cpu')
    layer.weight.fill_(1);layer.weight_scale.fill_(1.)
    tracker=FiniteTracker(debug=True);install_safe_operations(layer,tracker,True)
    assert layer._ps_fp32 and layer.weight.dtype==torch.int8
    x=torch.full((3,256),100000.)
    tracker.begin('cpu');got=layer(x);tracker.finish(got)
    ref=torch.nn.functional.linear(x,dequantize_rows(layer.weight,layer.weight_scale,True,256,dtype=torch.float32))
    torch.testing.assert_close(got,ref,atol=0,rtol=0)
