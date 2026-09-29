"""Native H3 instance attention CPU regression; не CUDA provider certification."""
import torch
import pytest
from powershard.config import DistributedConfig
from powershard.patch_config import H3PatchConfig
from powershard.attention import install_attention,configure_safe_qk
from powershard.fp16_safe import apply_fp16_safe


def test_safe_sdpa_real_h3_attention_and_counters(h3_factory,tmp_path):
    from safetensors.torch import save_file
    from powershard.checkpoint import Checkpoint
    net=h3_factory();reference=h3_factory()
    # Высокий V требует scaling, остаточный поток > half maximum не теряется.
    for model in (net,reference):
        model.blocks[0].attn.q_norm.weight.fill_(30.)
        model.blocks[0].attn.k_norm.weight.fill_(20.)
        model.blocks[0].attn.qkv_proj.weight.fill_(.3)
    path=tmp_path/'norms.safetensors';save_file(net.state_dict(),str(path))
    install_attention(net,DistributedConfig(attention_backend='sdpa'))
    configure_safe_qk(net,Checkpoint(path))
    apply_fp16_safe(net,H3PatchConfig(enabled=True));apply_fp16_safe(reference,H3PatchConfig(enabled=True))
    x=torch.full((5,32),1e5)
    with torch.no_grad():
        actual=net.blocks[0].attn(x);expected=reference.blocks[0].attn(x)
    assert actual.dtype==torch.float32 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected,atol=100.,rtol=.01)
    assert net.blocks[0].attn._ps_qk_scales != (1.,1.)
    assert net.blocks[0].attn._ps_attention.report()['call_counts']=={'dit:sdpa':1}
    assert reference.blocks[0].attn._ps_attention.report()['call_counts']=={'dit:math':1}
    assert net.token_refiner.blocks[0].attn._ps_attention_group=='token_refiner'
