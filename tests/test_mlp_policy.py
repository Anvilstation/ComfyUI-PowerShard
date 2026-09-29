import types
import torch
import pytest
from powershard.operations import Linear, Int8Linear, prepared_linears
from powershard.fp16_safe import chunked_mlp_forward, FiniteTracker
from powershard.patch_config import H3PatchConfig
from powershard.memory_policy import mlp_plan, plan_forward


def mlp(quant=False):
    torch.manual_seed(1729)
    block=torch.nn.Module()
    block.fc1=Linear(16,64,bias=True,dtype=torch.float16)
    block.fc2=Linear(32,16,bias=True,dtype=torch.float16)
    for m in (block.fc1,block.fc2):
        torch.nn.init.normal_(m.weight,std=.1)
        torch.nn.init.normal_(m.bias,std=.1)
    if quant:
        for name in ("fc1","fc2"):
            src=getattr(block,name)
            q=Int8Linear(src,dict(convrot=True,group_size=16),8)
            q.weight=torch.nn.Parameter(torch.randint(-64,64,src.weight.shape,dtype=torch.int8),requires_grad=False)
            q.weight_scale=torch.nn.Parameter(torch.full((src.out_features,1),.005),requires_grad=False)
            setattr(block,name,q)
    tracker=FiniteTracker()
    for i,m in enumerate(block.modules()):
        m._ps_safe=True;m._ps_fp32=False;m._ps_tracker=tracker;m._ps_finite_slot=tracker.register(str(i))
    block._ps_memory_context={"mlp_budget_bytes":2**30}
    block.forward=types.MethodType(chunked_mlp_forward,block)
    return block


@pytest.mark.parametrize("quant",[False,True])
@pytest.mark.parametrize("mode,chunk",[("manual",1),("manual",4),("manual",1024),("manual",4096),("manual",8192),("manual",16384),("off",512),("auto",512)])
def test_mlp_modes_match_unchunked(quant,mode,chunk):
    m=mlp(quant);x=torch.randn(2,9,16)*100000
    m._ps_mlp_policy=H3PatchConfig(True,True,False,512,"off");m._ps_mlp_chunk=512
    with torch.no_grad():
        ref=m(x)
        m._ps_mlp_policy=H3PatchConfig(True,True,False,chunk,mode);m._ps_mlp_chunk=chunk
        out=m(x)
    assert out.dtype==torch.float32 and torch.isfinite(out).all()
    torch.testing.assert_close(out,ref,rtol=.003,atol=16)
    assert not any(hasattr(v,"_ps_prepared") for v in m.modules())
    if quant:
        assert m.fc1.weight.dtype==torch.int8 and m.fc1.convrot and m.fc1.group_size==16


def test_preparation_is_once_per_active_mlp(monkeypatch):
    import powershard.operations as ops
    m=mlp(True);m._ps_mlp_policy=H3PatchConfig(True,True,False,2);m._ps_mlp_chunk=2
    count=0;original=ops.dequantize_rows
    def counted(*a,**kw):
        nonlocal count
        count+=1
        return original(*a,**kw)
    monkeypatch.setattr(ops,"dequantize_rows",counted)
    with torch.no_grad():m(torch.randn(17,16))
    assert count==64//8+16//8 # ten row tiles, not ten * nine token chunks
    assert m._ps_mlp_report["chunks"]==9


def test_preparation_releases_after_error():
    m=mlp()
    with pytest.raises(RuntimeError):
        with prepared_linears((m.fc1,m.fc2),2**30):
            assert m.fc1._ps_prepared
            raise RuntimeError("injected")
    assert not hasattr(m.fc1,"_ps_prepared")


def test_memory_estimate_never_rejects_manual():
    assert mlp_plan(20000,5376,14336,"off",512,0)["effective_tokens"]==20000
    assert mlp_plan(20000,5376,14336,"manual",16384,0)["effective_tokens"]==16384
    assert mlp_plan(20000,5376,14336,"auto",512,0)["effective_tokens"]==1
    assert plan_forward(1,2,3,4,5,2,"auto")["effective_prefetch"]==0
    assert H3PatchConfig(**{"mlp_chunk_tokens":8192}).mlp_chunk_mode=="manual"
    with pytest.raises(ValueError):H3PatchConfig(mlp_chunk_tokens=0)
