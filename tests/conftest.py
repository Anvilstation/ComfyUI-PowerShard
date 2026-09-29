import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import pytest


def pytest_addoption(parser):
    parser.addoption("--comfy", default=None, help="Путь к установленной ComfyUI для native integration tests")


@pytest.fixture(scope="session")
def h3_factory(request):
    path=request.config.getoption("--comfy")
    if path is None:
        pytest.skip("NOT_RUN: передайте --comfy для native H3 tests")
    sys.path.insert(0,str(Path(path).resolve()))
    saved=sys.argv
    sys.argv=["powershard-test","--cpu","--disable-dynamic-vram","--use-pytorch-cross-attention"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        from comfy.ldm.minimax.model import MiniMaxH3Model
    finally:
        sys.argv=saved
    import torch
    from powershard.operations import Operations
    from powershard.attention import install_attention
    from powershard.config import DistributedConfig
    def create(**overrides):
        config=dict(hidden_size=32,num_layers=2,token_refiner_num_layers=1,num_attention_heads=7,
                    attention_head_dim=8,ffn_hidden_size=64,text_dim=24,time_embed_dim=4,
                    adaln_curve_grid=17,rope_inv_freq_len=1,latents_dim=24,audio_latents_dim=32)
        config.update(overrides)
        with torch.device("meta"):
            net=MiniMaxH3Model(**config,dtype=torch.float16,device="meta",operations=Operations)
        net.to_empty(device="cpu").eval().requires_grad_(False)
        g=torch.Generator().manual_seed(40)
        for name,p in net.named_parameters():
            p.copy_(torch.ones_like(p) if "norm" in name else torch.randn(p.shape,generator=g).to(p.dtype)*.01)
        net.adaln_t_table.fill_(.1);net.rope.inv_freq.fill_(.01)
        install_attention(net,DistributedConfig())
        net._test_config=config
        return net
    return create
