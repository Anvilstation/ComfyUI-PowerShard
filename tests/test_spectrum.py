"""CPU algorithm/native sampler contracts. НЕ CUDA/NCCL/FSDP evidence."""
from types import SimpleNamespace
import torch
import pytest
from powershard.spectrum import SpectrumEngine, SpectrumBlockGate,forecast_weights
from powershard.spectrum_config import SpectrumConfig
from powershard.conditioning_cache import content_hash


def metadata(i,branch="a",run="run",steps=10):
    return dict(run_id=run,eligible=True,steps=steps,step_index=i,coordinate=1-i*.2,conditioning_key=branch)


def test_linear_forecast_and_hash_framing():
    points=[1.,.8,.6];w=forecast_weights(points,.4,2,.001,0.)
    values=torch.tensor([[x*2+7,x*3-1] for x in points])
    torch.testing.assert_close((values*torch.tensor(w)[:,None]).sum(0),torch.tensor([7.8,.2]))
    assert content_hash([1,23])!=content_hash([12,3])
    assert content_hash({"ab":"c"})!=content_hash({"a":"bc"})
    assert content_hash([1])!=content_hash((1,))


def test_gates_skip_child_call_keep_targets_and_bound_history():
    cfg=SpectrumConfig(enabled=True,blend=0.,history_mib=1)
    engine=SpectrumEngine(cfg)
    layout=SimpleNamespace(segments=[(0,2,"text"),(2,4,"ref_img"),(4,6,"audio"),(6,11,"video")])
    class Block(torch.nn.Module):
        def forward(self,h,**kwargs):return h+2
    child=Block();calls=[]
    child.register_forward_pre_hook(lambda m,a:calls.append(1))
    gate=SpectrumBlockGate(child,engine,0,1)
    for i in range(3):
        engine.begin(metadata(i))
        h=torch.full((11,4),float(i))
        gate(h,transformer_options={"minimax_h3_layout":layout})
        h.fill_(12345) # history должна владеть отдельными snapshots
    engine.begin(metadata(3));assert engine.forecast
    out=gate(torch.full((11,4),3.),transformer_options={"minimax_h3_layout":layout})
    assert len(calls)==3 # __call__ и prehook child реально не вызваны
    torch.testing.assert_close(out[:4],torch.full((4,4),3.))
    torch.testing.assert_close(out[4:],torch.full((7,4),5.))
    assert engine.bytes==3*7*4*4
    engine.begin(metadata(4));assert not engine.forecast and engine.reason=="actual_refresh"
    engine.begin(metadata(3,branch="negative"));assert not engine.forecast
    engine.begin(metadata(0,run="new"));assert not engine.entries and engine.bytes==0


def test_insufficient_history_budget_is_actual():
    e=SpectrumEngine(SpectrumConfig(enabled=True,history_mib=0))
    layout=SimpleNamespace(segments=[(0,1,"audio"),(1,4,"video")])
    for i in range(5):
        e.begin(metadata(i));assert not e.forecast;e.capture(torch.ones(4,2),layout)
    assert e.bytes==0 and not e.entries


def test_sampler_capability_and_few_steps(h3_factory):
    import comfy.samplers
    from powershard.spectrum_host import sampler_capability
    assert sampler_capability(comfy.samplers.sampler_object('euler'))[0]
    assert not sampler_capability(comfy.samplers.sampler_object('heun'))[0]
    e=SpectrumEngine(SpectrumConfig(enabled=True))
    layout=SimpleNamespace(segments=[(0,2,'audio'),(2,5,'video')])
    for i in range(4):
        e.begin(metadata(i,steps=4));assert not e.forecast;e.capture(torch.ones(5,3),layout)
    assert e.stats['actual']==4 and e.stats['forecast']==0


@pytest.mark.parametrize("world",[1,2,3,4,6,9])
def test_local_history_reconstructs_uneven_targets(world):
    h=torch.arange(13*3).reshape(13,3).float();segments=((5,8,"audio"),(8,13,"video"))
    pieces=[SpectrumEngine(rank=r,world=world)._local_target(h,segments)[0] for r in range(world)]
    torch.testing.assert_close(torch.cat(pieces),h[5:])
    assert sum(p.numel() for p in pieces)==8*3


def test_native_sampler_worker_spectrum(h3_factory,tmp_path,request):
    from safetensors.torch import save_file
    import comfy.sample,comfy.nested_tensor,comfy.supported_models
    from powershard.comfy_adapter import RemoteH3,DiffusionProxy,PowerShardPatcher
    from powershard.nodes import PowerShardH3FP16Patcher,PowerShardSpectrum
    from powershard.config import DistributedConfig
    from test_native_pipeline import CPUContractSession
    net=h3_factory();ck=tmp_path/"h3.safetensors";save_file(net.state_dict(),str(ck))
    session=CPUContractSession(str(ck),DistributedConfig(release_after_sampling=False),request.config.getoption("--comfy"),tmp_path/"reports")
    mc=comfy.supported_models.MiniMaxH3(dict(net._test_config,image_model="minimax_h3",disable_unet_model_creation=True,dtype=torch.float16))
    mc.manual_cast_dtype=torch.float16
    host=RemoteH3(mc,device=torch.device("cpu"));host.diffusion_model=DiffusionProxy(session,net._test_config)
    base=PowerShardPatcher(host,torch.device("cpu"),torch.device("cpu"),size=1)
    safe=PowerShardH3FP16Patcher().patch(base)[0]
    patched=PowerShardSpectrum().patch(safe,enabled=True,blend=0.)[0]
    removed=PowerShardSpectrum().patch(patched,enabled=False)[0]
    assert "spectrum" not in safe.session.role_options
    assert not removed.session.role_options["spectrum"]["enabled"]
    assert list(removed.wrappers["outer_sample"])==["powershard_run"]
    latent=comfy.nested_tensor.NestedTensor([torch.zeros(1,24,2,4,4),torch.zeros(1,32,2,5)])
    conditioning=[[torch.ones(1,5,24),{}]]
    try:
        from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced,BasicGuider,BasicScheduler,RandomNoise,KSamplerSelect
        outputs=[]
        for _ in range(2):
            with torch.no_grad():
                out=SamplerCustomAdvanced.execute(RandomNoise.execute(5)[0],BasicGuider.execute(patched,conditioning)[0],
                    KSamplerSelect.execute("euler")[0],BasicScheduler.execute(patched,"simple",7,1.)[0],{"samples":latent})[0]
            outputs.append(out["samples"].unbind())
            assert all(torch.isfinite(t).all() for t in outputs[-1])
        finals=[x["ranks"][0]["spectrum_end_run"] for x in patched.session.history if x.get("command")=="end_run"]
        assert len(finals)==2 and finals[0]["run_id"]!=finals[1]["run_id"]
        assert all(x["counters"]["forecast"]>=1 and x["counters"]["actual"]>=4 for x in finals)
        for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert len(list((tmp_path/"reports").glob("run-*.json")))==2
    finally:
        for m in (base,safe,patched,removed):m.session.close()
