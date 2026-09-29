import json
from pathlib import Path
import struct
import subprocess
import sys
import pytest
import torch
from powershard.config import DistributedConfig, shard_bounds
from powershard.checkpoint import Checkpoint, infer_h3_config
from powershard.attention import exact_attention, rope_split_half
from powershard.operations import regular_hadamard, dequantize_rows, Int8Linear, Linear
from powershard.wire import write_payload, read_payload, validate_options


def test_import_without_torch():
    code = "import sys; from powershard.nodes import NODE_CLASS_MAPPINGS as n; assert {'PowerShardConfig','PowerShardH3Loader','PowerShardH3FP16Patcher','PowerShardH3TextEncoder','PowerShardRelease','PowerShardDiagnostics'} <= n.keys(); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, '-S', '-c', code], check=True, cwd=Path(__file__).resolve().parents[1])


@pytest.mark.parametrize('rows',[0,1,2,3,4,7,56,5376,21504,28672])
@pytest.mark.parametrize('world',[1,2,3,4,6,11])
def test_partition(rows,world):
    slices=[shard_bounds(rows,r,world) for r in range(world)]
    assert slices[0][0]==0 and slices[-1][1]==rows
    assert all(slices[i][1]==slices[i+1][0] for i in range(world-1))
    x=torch.arange(rows)
    assert torch.equal(torch.cat([x[a:b] for a,b in slices]),x)


def test_config():
    for ids in [(),('',),('bad',),('all','1')]:
        with pytest.raises(ValueError):DistributedConfig(ids)
    with pytest.raises(ValueError):DistributedConfig(backend='ddp')


@pytest.mark.parametrize('length',[1,7,13,49])
def test_attention(length):
    g=torch.Generator().manual_seed(10)
    q=torch.randn(7,length,8,generator=g)
    k=torch.randn(7,length+3,8,generator=g)
    v=torch.randn(7,length+3,8,generator=g)
    expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    actual=exact_attention(q,k,v,query_chunk=4,key_chunk=5)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    # Нечётное число heads и tokens: query splitting не требует heads%3 == 0.
    merged=torch.cat([exact_attention(q[:,a:b],k,v,4,5) for a,b in [shard_bounds(length,r,3) for r in range(3)]],dim=1)
    torch.testing.assert_close(merged,expected,rtol=2e-5,atol=2e-6)


def test_convrot():
    h=regular_hadamard(256,'cpu')
    torch.testing.assert_close(h@h.T,torch.eye(256),rtol=0,atol=0)
    g=torch.Generator().manual_seed(5)
    q=torch.randint(-100,100,(13,512),generator=g,dtype=torch.int8)
    scale=torch.rand((13,1),generator=g)*.001
    actual=dequantize_rows(q,scale,True,256)
    # Независимое сравнение с audited Comfy Kitchen eager implementation.
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    expected=dequantize_int8_convrot_weight(q,scale,256).half()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    src=Linear(512,13,bias=False,device='meta',dtype=torch.float16)
    layer=Int8Linear(src,dict(convrot=True,group_size=256),rows=4).to_empty(device='cpu')
    layer.weight.data.copy_(q);layer.weight_scale.data.copy_(scale)
    x=torch.randn(3,512,generator=g).half()
    torch.testing.assert_close(layer(x),torch.nn.functional.linear(x,expected),atol=1e-3,rtol=1e-3)
    assert layer.weight.dtype==torch.int8 and not layer.weight.requires_grad
    assert not any(p.dtype==torch.float16 and p.shape==q.shape for p in layer.parameters())


def test_overflow():
    q=torch.full((4,256),127,dtype=torch.int8)
    with pytest.raises(FloatingPointError):dequantize_rows(q,torch.full((4,1),1e8),False,256)


def test_wire(tmp_path):
    value={'args':(torch.arange(11),[torch.randn(3,4)]),'kwargs':{'transformer_options':{'sample_sigmas':torch.tensor([1.,0.])},'seed':99}}
    write_payload(tmp_path,value);restored=read_payload(tmp_path)
    assert isinstance(restored['args'],tuple)
    torch.testing.assert_close(restored['args'][0],value['args'][0],rtol=0,atol=0)
    assert restored['kwargs']['seed']==99
    with pytest.raises(TypeError):write_payload(tmp_path,{'callback':lambda:None})
    with pytest.raises(ValueError):validate_options({'patches_replace':{'dit':{'foo':1}}})


def test_checkpoint_header_validation(tmp_path):
    from safetensors.torch import save_file
    p=tmp_path/'test.safetensors';save_file({'x':torch.randn(5,7)},str(p))
    c=Checkpoint(p);assert c.tensors['x']['shape']==[5,7]
    with p.open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError):Checkpoint(p)


def test_real_metadata():
    root=Path(__file__).resolve().parents[1]
    for variant in ('fl2va','ref2va'):
        header=json.loads((root/f'models/minimax_h3_{variant}_pruned_int8_convrot.safetensors.header.json').read_text())
        cfg=infer_h3_config(header)
        assert cfg['num_layers']==50 and cfg['num_attention_heads']==56
        assert cfg['time_embed_dim']==8 and cfg['token_refiner_num_layers']==2
        assert header['blocks.0.attn.qkv_proj.weight']['dtype']=='I8'


def test_native_wrapper_stripped(tmp_path):
    from powershard.host_guard import guard_sampling
    value={'transformer_options':{'wrappers':{'prepare_sampling':{'powershard':[guard_sampling]}}}}
    write_payload(tmp_path,value)
    assert read_payload(tmp_path)['transformer_options']['wrappers']['prepare_sampling']['powershard']==[]
    with pytest.raises(ValueError):
        write_payload(tmp_path,{'transformer_options':{'wrappers':{'prepare_sampling':{'powershard':[lambda:None]}}}})
