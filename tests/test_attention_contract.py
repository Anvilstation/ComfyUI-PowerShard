import sys
from types import SimpleNamespace
import pytest
import torch
from powershard.attention_contract import AttentionOptions as O,math_attention,sdpa_attention,UnsupportedAttention
from powershard.vllm_adapter import VLLMFlashAdapter
from powershard.attention_policy import AttentionDispatcher
from powershard.config import DistributedConfig


def tensors(lq=5,lk=7,hq=4,hk=2,d=8,dv=8,dtype=torch.float32):
    g=torch.Generator().manual_seed(9)
    return tuple(torch.randn(shape,generator=g,dtype=dtype)[...,::2] for shape in
        [(2,lq,hq,d*2),(2,lk,hk,d*2),(2,lk,hk,dv*2)])


def reference(q,k,v,o):
    # Независимая dense reference, не вызывает bias_tile / provider adapter.
    q,k,v=(x.float().transpose(1,2) for x in (q,k,v))
    k=k.repeat_interleave(q.shape[1]//k.shape[1],1);v=v.repeat_interleave(q.shape[1]//v.shape[1],1)
    a=q.shape[-2];b=k.shape[-2];offset=b-a if o.causal_alignment=='bottom_right' else 0
    i=torch.arange(a)[:,None]+offset;j=torch.arange(b)[None,:]
    scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-.5 if o.softmax_scale is None else o.softmax_scale)
    if o.causal:scores.masked_fill_(j>i,-torch.inf)
    if o.window_size[0]>=0:scores.masked_fill_(j<i-o.window_size[0],-torch.inf)
    if o.window_size[1]>=0:scores.masked_fill_(j>i+o.window_size[1],-torch.inf)
    if o.alibi_slopes is not None:scores-=o.alibi_slopes.reshape(1,-1,1,1)*(i-j).abs()
    if o.mask is not None:
        if o.mask.dtype==torch.bool:scores.masked_fill_(~o.mask,-torch.inf)
        else:scores+=o.mask
    p=torch.softmax(scores,-1)
    # Тесты с fully masked rows имеют отдельный явно заданный expected.
    return (p@v).transpose(1,2)


@pytest.mark.parametrize('scale',[.0,.13,.7,-.1])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('alignment',['upper_left','bottom_right'])
def test_math_rectangular_gqa(scale,causal,alignment):
    q,k,v=tensors()
    o=O(softmax_scale=scale,causal=causal,causal_alignment=alignment)
    out=math_attention(q,k,v,o,2,3)
    assert out.shape==q.shape and out.dtype==q.dtype and torch.isfinite(out).all()
    torch.testing.assert_close(out,reference(q,k,v,o),atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('kind',['bool','additive','window','alibi'])
def test_masks_windows_alibi(kind):
    q,k,v=tensors(dv=6)
    kwargs={}
    if kind=='bool':kwargs['mask']=torch.arange(7)[None,:] != 2
    if kind=='additive':kwargs['mask']=torch.linspace(-.7,.5,7)[None,:]
    if kind=='window':kwargs['window_size']=(2,1)
    if kind=='alibi':kwargs['alibi_slopes']=torch.tensor([.1,.2,.3,.4])
    o=O(softmax_scale=.29,**kwargs)
    torch.testing.assert_close(math_attention(q,k,v,o,2,3),reference(q,k,v,o),atol=2e-6,rtol=2e-5)


def fake_module(calls):
    def varlen(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,
               dropout_p=0.,softmax_scale=None,causal=False,window_size=(-1,-1),alibi_slopes=None,
               deterministic=False,return_attn_probs=False):
        calls.append(dict(scale=softmax_scale,cuq=cu_seqlens_q.tolist(),cuk=cu_seqlens_k.tolist(),
            maxq=max_seqlen_q,maxk=max_seqlen_k,dropout=dropout_p,deterministic=deterministic))
        outputs=[]
        for a,b,c,d in zip(cu_seqlens_q[:-1],cu_seqlens_q[1:],cu_seqlens_k[:-1],cu_seqlens_k[1:]):
            o=O(softmax_scale=softmax_scale,causal=causal,causal_alignment='bottom_right',window_size=window_size,alibi_slopes=alibi_slopes,dropout_p=dropout_p)
            outputs.append(math_attention(q[a:b][None],k[c:d][None],v[c:d][None],o)[0])
        # Сборка, которая возвращает extra result даже без return flag.
        return torch.cat(outputs),None
    return SimpleNamespace(flash_attn_varlen_func=varlen,__file__=__file__)


@pytest.mark.parametrize('scale',[.12,.9])
@pytest.mark.parametrize('hk',[1,2,4])
def test_custom_dense_cross_gqa_batch_noncontiguous(scale,hk):
    calls=[];adapter=VLLMFlashAdapter(fake_module(calls));q,k,v=tensors(hk=hk,dtype=torch.float16)
    before=list(sys.path);old=sys.modules.get('flash_attn')
    out=adapter(q,k,v,O(softmax_scale=scale))
    torch.testing.assert_close(out.float(),reference(q,k,v,O(softmax_scale=scale)),atol=.003,rtol=.003)
    assert calls[0]['scale']==scale and calls[0]['cuq']==[0,5,10] and calls[0]['cuk']==[0,7,14]
    assert calls[0]['maxq']==5 and calls[0]['maxk']==7
    assert sys.path==before and sys.modules.get('flash_attn') is old


def test_unequal_varlen_batch():
    q,k,v=tensors(dtype=torch.float16);calls=[];adapter=VLLMFlashAdapter(fake_module(calls))
    qp=torch.cat((q[0,:3],q[1,:5]));kp=torch.cat((k[0,:6],k[1,:4]));vp=torch.cat((v[0,:6],v[1,:4]))
    o=O(softmax_scale=.31)
    out=adapter.packed(qp,kp,vp,lengths_q=(3,5),lengths_k=(6,4),options=o)
    expected=torch.cat((reference(q[:1,:3],k[:1,:6],v[:1,:6],o)[0],reference(q[1:,:5],k[1:,:4],v[1:,:4],o)[0]))
    torch.testing.assert_close(out.float(),expected,atol=.003,rtol=.003)
    assert calls[0]['cuq']==[0,3,8] and calls[0]['cuk']==[0,6,10]
    with pytest.raises(ValueError):adapter.packed(qp,kp,vp,lengths_q=(3,6),lengths_k=(6,4))


def test_unsupported_scale_not_silently_dropped():
    def legacy(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,causal=False):raise AssertionError('kernel must not run')
    adapter=VLLMFlashAdapter(SimpleNamespace(flash_attn_varlen_func=legacy,__file__=__file__))
    with pytest.raises(UnsupportedAttention,match='softmax_scale'):adapter(*tensors(dtype=torch.float16),O(softmax_scale=.2))


def test_fallback_rectangular_mask_return_and_strict():
    dispatcher=AttentionDispatcher(DistributedConfig(attention_backend='math'))
    dispatcher.effective='vllm_flash_attn';dispatcher.provider=VLLMFlashAdapter(fake_module([]))
    q,k,v=tensors(dtype=torch.float16)
    for o in (O(causal=True),O(mask=torch.ones(5,7,dtype=torch.bool)),O(return_attn_probs=True)):
        with pytest.warns(UserWarning):out=dispatcher(q,k,v,o)
        if o.return_attn_probs:
            assert len(out)==3 and out[1].shape==(2,4,5) and out[2].shape==(2,4,5,7)
            torch.testing.assert_close(out[2].sum(-1),torch.ones(2,4,5))
            out=out[0]
        torch.testing.assert_close(out.float(),reference(q,k,v,o),atol=.003,rtol=.003)
    dispatcher.config=DistributedConfig(allow_fallback=False)
    with pytest.raises(UnsupportedAttention):dispatcher(q,k,v,O(causal=True))


def test_dropout_forwarded_and_unknown_kwargs():
    calls=[];adapter=VLLMFlashAdapter(fake_module(calls));q,k,v=tensors(dtype=torch.float16)
    adapter(q,k,v,O(dropout_p=.25,deterministic=True))
    assert calls[0]['dropout']==.25 and calls[0]['deterministic']
    with pytest.raises(TypeError):O(unknown=True)


def test_fully_masked_rows_zero_but_nan_not_hidden():
    q,k,v=tensors();mask=torch.ones(5,7,dtype=torch.bool);mask[0]=False
    out=math_attention(q,k,v,O(mask=mask),2,3)
    assert torch.isfinite(out).all() and torch.count_nonzero(out[:,0])==0
    q[0,1,0,0]=torch.nan
    assert torch.isnan(math_attention(q,k,v,O(),2,3)).any()


def test_fatal_runtime_not_fallback():
    d=AttentionDispatcher(DistributedConfig());d.effective='vllm_flash_attn'
    def fail(*args):raise RuntimeError('CUDA illegal memory access')
    d.provider=fail
    with pytest.raises(RuntimeError,match='illegal memory'):d(*tensors(),O())


def test_sdpa_scale_reference():
    q,k,v=tensors(hk=1)
    torch.testing.assert_close(sdpa_attention(q,k,v,O(softmax_scale=.81)),reference(q,k,v,O(softmax_scale=.81)),atol=2e-6,rtol=2e-5)
