"""Настоящий subprocess pipe transport и native sampler; CPU small weights, НЕ FSDP/CUDA."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import pytest
import torch
from safetensors.torch import save_file
from powershard.config import DistributedConfig
from powershard.patch_config import H3PatchConfig
from powershard.runtime import Session


class CPUContractSession(Session):
    def with_patch(self, patch):
        return CPUContractSession(self.checkpoint, self.config, self.comfy_path, self.report_dir, patch=patch,role_options=self.role_options)

    def with_spectrum(self, spectrum):
        return CPUContractSession(self.checkpoint,self.config,self.comfy_path,self.report_dir,patch=self.patch,
                                  role_options=dict(self.role_options,spectrum=spectrum.to_dict()))

    def start(self, cancel=None):
        if self.running:return
        self.close()
        self.path=Path(tempfile.mkdtemp(prefix="powershard-test-cpu-"))
        self.report_dir.mkdir(parents=True,exist_ok=True)
        settings={"checkpoint":self.checkpoint,"patch":self.patch.to_dict(),"comfy_path":self.comfy_path,"role_options":self.role_options}
        (self.path/"settings.json").write_text(json.dumps(settings))
        log=(self.report_dir/(self.path.name+".log")).open("w")
        env=os.environ.copy();env["OMP_NUM_THREADS"]="1"
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name("cpu_contract_worker.py")),str(self.path/"settings.json")],
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
        self.processes=[proc];self.logs=[log];self.responses=[queue.Queue()]
        threading.Thread(target=self._reader,args=(proc.stdout,self.responses[0]),daemon=True).start()
        try:self._wait(0,cancel)
        except BaseException:self.close();raise


def test_worker_patch_extra_conds_and_sampler(h3_factory,tmp_path,request,monkeypatch):
    import comfy.sample,comfy.samplers,comfy.nested_tensor,comfy.supported_models
    from powershard.comfy_adapter import RemoteH3,DiffusionProxy,PowerShardPatcher
    from powershard.nodes import PowerShardH3FP16Patcher
    net=h3_factory();net.condition_proj.weight.fill_(1.)
    ck=tmp_path/"tiny-h3.safetensors";save_file(net.state_dict(),str(ck))
    from powershard.comfy_adapter import load_model
    # CPU contract fixture, не GPU inventory / FSDP test.
    monkeypatch.setattr('powershard.devices.visible_inventory',lambda:[dict(user_id=str(i),uuid=f'GPU-{i}',name='CPU mock',total_memory=0) for i in range(3)])
    gpu_budget=load_model(str(ck),DistributedConfig(cpu_offload=False))
    cpu_budget=load_model(str(ck),DistributedConfig(cpu_offload=True))
    assert gpu_budget.model_size()-cpu_budget.model_size()==gpu_budget.powershard_memory_plan['shard_bytes_lower_bound']
    assert not gpu_budget.session.running and not cpu_budget.session.running
    config=DistributedConfig(reserve_gib=0,allow_unverified=True,release_after_sampling=False,timeout_s=30)
    session=CPUContractSession(str(ck),config,request.config.getoption("--comfy"),tmp_path/"workers")
    mc=comfy.supported_models.MiniMaxH3(dict(net._test_config,image_model="minimax_h3",disable_unet_model_creation=True,dtype=torch.float16))
    mc.manual_cast_dtype=torch.float16
    host=RemoteH3(mc,device=torch.device("cpu"));host.diffusion_model=DiffusionProxy(session,net._test_config)
    source=PowerShardPatcher(host,torch.device("cpu"),torch.device("cpu"),size=1)
    patched=PowerShardH3FP16Patcher().patch(source,True,True,True,3)[0]
    disabled=PowerShardH3FP16Patcher().patch(patched,False,True,False)[0]
    assert source.model is not patched.model
    assert source.model.diffusion_model is not patched.model.diffusion_model
    assert source.session is not patched.session
    assert not source.session.patch.active and not disabled.session.patch.active
    assert patched.session.patch.active
    assert not list(patched.model.diffusion_model.parameters())
    text=torch.full((1,5,24),1e5)
    try:
        with pytest.raises(RuntimeError,match="non-finite|Переполнение|NaN/Inf"):
            source.model.extra_conds(cross_attn=text,device=torch.device("cpu"))
        assert not source.session.running
        conds=patched.model.extra_conds(cross_attn=text,device=torch.device("cpu"))
        hidden=conds["c_crossattn"].cond
        assert hidden.shape==(1,5,32) and hidden.dtype==torch.float32 and torch.isfinite(hidden).all()
        worker_pid=patched.session.processes[0].pid
        latent=comfy.nested_tensor.NestedTensor([torch.zeros(1,24,2,4,4),torch.zeros(1,32,2,5)])
        conditioning=[[text,{}]];outputs=[]
        for seed in (5,6,5):
            noise=comfy.sample.prepare_noise(latent,seed)
            with torch.no_grad():
                out=comfy.sample.sample(patched,noise,2,1.,"euler","simple",conditioning,conditioning,latent,seed=seed,disable_pbar=True)
            assert all(torch.isfinite(x).all() for x in out.unbind())
            outputs.append([x.clone() for x in out.unbind()])
        assert worker_pid==patched.session.processes[0].pid
        for x,y in zip(outputs[0],outputs[2]):torch.testing.assert_close(x,y,atol=0,rtol=0)
        assert not torch.equal(outputs[0][0],outputs[1][0])
        # Именно nodes_custom_sampler -> CFGGuider -> patcher_extension,
        # включая host callback/x0 и оба H3 audio/video latent outputs.
        from comfy_extras.nodes_custom_sampler import (SamplerCustomAdvanced, BasicGuider,
                                                       BasicScheduler, RandomNoise, KSamplerSelect)
        guider=BasicGuider.execute(patched,conditioning)[0]
        sigmas=BasicScheduler.execute(patched,"simple",2,1.)[0]
        sampled=SamplerCustomAdvanced.execute(RandomNoise.execute(5)[0],guider,
                    KSamplerSelect.execute("euler")[0],sigmas,{"samples":latent})
        for result in sampled:
            assert all(torch.isfinite(x).all() for x in result["samples"].unbind())
        assert worker_pid==patched.session.processes[0].pid
        changed=patched.model.extra_conds(cross_attn=torch.ones(1,9,24)*-1e5,device=torch.device('cpu'))['c_crossattn'].cond
        assert changed.shape==(1,9,32) and not torch.equal(hidden,changed[:,:5])
        assert worker_pid==patched.session.processes[0].pid
        clone=patched.clone();clone.model_options["transformer_options"]["test"]="isolated"
        assert "test" not in patched.model_options["transformer_options"]
        with pytest.raises(RuntimeError,match="non-finite|Переполнение|NaN/Inf"):
            disabled.model.extra_conds(cross_attn=text,device=torch.device("cpu"))
        # Ошибка другой session не загрязняет процесс/patch этого CPU test.
        assert patched.session.call("preprocess_text",(text,),{}).dtype==torch.float32
    finally:
        for item in (source,patched,disabled):item.session.close()


def test_capabilities_accept_current_and_missing_method(h3_factory,request,monkeypatch):
    from powershard.source_guard import verify_comfy
    import comfy.ldm.minimax.model as mm
    report=verify_comfy("/path-is-not-used-for-version-gating")
    assert report["admission"]=="capabilities_only"
    with monkeypatch.context() as m:
        m.delattr(mm.MiniMaxH3Model,"preprocess_text_embeds")
        with pytest.raises(RuntimeError,match="preprocess_text_embeds"):verify_comfy("ignored")
