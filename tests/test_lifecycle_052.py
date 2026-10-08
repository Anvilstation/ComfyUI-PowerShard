"""CPU lifecycle/protocol regressions. No CUDA/FSDP certification."""
import io
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import pytest
from powershard.config import DistributedConfig
from powershard.patch_config import H3PatchConfig
from powershard.runtime import Session,reusable_session,release_all


def test_mlp_disabled_at_every_new_entrypoint():
    from powershard.qwen import QwenConfig
    from powershard.nodes import PowerShardH3MLP,PowerShardQwenMLP,PowerShardH3QwenLoader
    import inspect
    assert H3PatchConfig().mlp_chunk_mode==QwenConfig().mlp_chunk_mode=="off"
    assert inspect.signature(PowerShardH3QwenLoader.load).parameters["mlp_chunk_mode"].default=="off"
    for node in (PowerShardH3MLP,PowerShardQwenMLP):
        assert node.INPUT_TYPES()["required"]["mode"][1]["default"]=="off"
        assert inspect.signature(node.patch).parameters["mode"].default=="off"


def test_h3_loader_keep_switch_controls_actual_session(tmp_path,monkeypatch):
    import sys
    from powershard.nodes import PowerShardH3Loader
    calls=[]
    monkeypatch.setitem(sys.modules,"folder_paths",SimpleNamespace(
        get_filename_list=lambda kind:["tiny.safetensors"],
        get_full_path_or_raise=lambda *args:str(tmp_path/"tiny.safetensors"),
        get_output_directory=lambda:str(tmp_path)))
    # Avoid native import; verify the real config passed across the node boundary.
    monkeypatch.setitem(sys.modules,"powershard.comfy_adapter",SimpleNamespace(load_model=lambda *a:calls.append(a) or a[1]))
    cls=PowerShardH3Loader
    assert cls.INPUT_TYPES()["optional"]["keep_in_memory"][1]["default"]
    assert not cls().load("tiny.safetensors",DistributedConfig())[0].release_after_sampling
    assert cls().load("tiny.safetensors",DistributedConfig(),False)[0].release_after_sampling
    assert len(calls)==2


@pytest.mark.parametrize("policy",["release","cpu_shards","keep"])
def test_native_qwen_clone_and_mlp_detach_after_load(h3_factory,tmp_path,policy):
    import torch
    import comfy.model_management as mm
    from powershard.qwen import QwenConfig
    from powershard.qwen_adapter import QwenProxy,QwenPatcher,DistributedQwenCLIP
    session=Session(None,DistributedConfig(weight_placement="cpu"),tmp_path,tmp_path,
                    role="qwen",role_options=QwenConfig(idle_policy=policy).to_dict())
    proxy=QwenProxy(session,QwenConfig(idle_policy=policy),"fixture")
    clip=DistributedQwenCLIP(no_init=True)
    clip.patcher=QwenPatcher(proxy,torch.device("cpu"),torch.device("cpu"),size=1)
    clip.cond_stage_model=proxy;clip.tokenizer=None;clip.layer_idx=None
    clip.tokenizer_options={};clip.use_clip_schedule=False;clip.apply_hooks_to_conds=None
    detached=[]
    session.deactivate=lambda:detached.append(True)
    candidates=[clip,clip.clone(),clip.with_mlp("manual",13)]
    for candidate in candidates:
        candidate.patcher.session.deactivate=lambda:detached.append(True)
        model=mm.LoadedModel(candidate.patcher)
        model.model_load()
        assert model.model_unload() # actual load_models_gpu -> free_memory path
        proxy=candidate.cond_stage_model
        assert proxy.model_lowvram is False and proxy.lowvram_patch_counter==0
        assert proxy.model_loaded_weight_memory==proxy.model_offload_buffer_memory==0
        assert proxy.current_weight_patches_uuid is None
    assert len(detached)==3


def test_identical_config_reuses_owner_changed_config_does_not(tmp_path):
    config=DistributedConfig(weight_placement="cpu",release_after_sampling=False)
    first=reusable_session("checkpoint",config,tmp_path,tmp_path)
    assert reusable_session("checkpoint",config,tmp_path,tmp_path) is first
    assert first.with_patch(first.patch) is first
    assert reusable_session("checkpoint",DistributedConfig(weight_placement="gpu",release_after_sampling=False),tmp_path,tmp_path) is not first
    assert first.with_patch(H3PatchConfig(enabled=True)) is not first
    no_keep=DistributedConfig(release_after_sampling=True)
    assert reusable_session(None,no_keep,tmp_path,tmp_path) is not reusable_session(None,no_keep,tmp_path,tmp_path)


def test_already_cancelled_prompt_does_not_start_rpc_or_unload_weights(tmp_path,monkeypatch):
    session=Session(None,DistributedConfig(weight_placement="cpu",release_after_sampling=False),tmp_path,tmp_path)
    calls=[]
    monkeypatch.setattr(session,"start",lambda cancel:calls.append("start"))
    monkeypatch.setattr(session,"close",lambda **kwargs:calls.append("close"))
    def cancel():raise InterruptedError()
    with pytest.raises(InterruptedError):session.call("forward",(),{},cancel)
    assert calls==[]


def test_cuda_failure_does_not_keep_a_damaged_session(tmp_path,monkeypatch):
    session=Session(None,DistributedConfig(weight_placement="cpu",release_after_sampling=False),tmp_path,tmp_path)
    session.path=tmp_path/"workers";session.path.mkdir()
    session.processes=[SimpleNamespace(stdin=io.StringIO(),poll=lambda:None)]
    monkeypatch.setattr(session,"start",lambda cancel:None)
    def failure(sequence,cancel):raise RuntimeError("CUDA device-side assert")
    monkeypatch.setattr(session,"_wait",failure)
    closed=[];monkeypatch.setattr(session,"close",lambda **kwargs:closed.append(True))
    with pytest.raises(RuntimeError,match="device-side assert"):session.call("forward",(),{})
    assert closed==[True] and not session.draining
    session.processes=[]


@pytest.mark.parametrize("role,policy,keep",[("h3",None,True),("h3",None,False),("qwen","cpu_shards",True),("qwen","keep",True),("qwen","release",False)])
def test_memory_pressure_retains_only_requested_cpu_shards(tmp_path,role,policy,keep):
    session=Session(None,DistributedConfig(weight_placement="cpu",release_after_sampling=not keep),tmp_path,tmp_path,
                    role=role,role_options={"idle_policy":policy} if policy else {})
    session.processes=[SimpleNamespace(poll=lambda:None)]
    calls=[];session.idle=lambda:calls.append("idle");session.close=lambda **kw:calls.append("close")
    session.deactivate()
    assert calls==["idle" if keep else "close"]
    session.processes=[]


@pytest.mark.parametrize("ranks",[1,3])
def test_cancel_drains_partial_rank_replies_and_preserves_inputs(tmp_path,monkeypatch,ranks):
    """Real _wait, resumable queues, phase locking and cancellation exception."""
    import powershard.runtime as runtime
    session=Session(None,DistributedConfig(weight_placement="cpu",release_after_sampling=False),tmp_path,tmp_path)
    session.path=tmp_path/"workers";session.path.mkdir()
    session.responses=[queue.Queue() for _ in range(ranks)]
    monkeypatch.setattr(session,"start",lambda cancel:None)
    monkeypatch.setattr(runtime,"_ACTIVE",session)
    released=threading.Event();received=threading.Event();inputs=[];outputs=[];commands=[]
    class Pipe(io.StringIO):
        def __init__(self,rank):super().__init__();self.rank=rank
        def write(self,value):
            req=json.loads(value);commands.append((self.rank,req["command"]))
            if req["command"]=="forward":
                inputs.append(Path(req["input"]));outputs.append(Path(req["output"]))
                if self.rank==0 and ranks>1:
                    session.responses[0].put(dict(sequence=req["sequence"],metrics={}))
                else:
                    def respond():
                        released.wait(5)
                        session.responses[self.rank].put(dict(sequence=req["sequence"],metrics={}))
                    threading.Thread(target=respond,daemon=True).start()
                received.set()
            elif req["command"] in ("end_run","idle"):
                session.responses[self.rank].put(dict(sequence=req["sequence"],phase_offload={"after":{"allocated":9}}))
            return len(value)
    session.processes=[SimpleNamespace(stdin=Pipe(i),poll=lambda:None) for i in range(ranks)]
    def cancel():
        # Let the first rank reply be consumed before interrupting a multi-rank wait.
        state=session._wait_state
        if received.is_set() and (ranks==1 or state and state["result"][0] is not None):
            raise InterruptedError("user cancellation")
    try:
        with pytest.raises(InterruptedError):session.call("forward",(),{},cancel)
        assert session.draining and session.running and all(p.is_dir() for p in inputs)
        session.finish_sampling() # sampler finally must not shut down/drain again
        assert not any(c=="shutdown" for _,c in commands)
        released.set()
        assert session._drain_done.wait(5)
        assert session.running and session.idle_on_cpu
        assert len([c for _,c in commands if c=="forward"])==ranks
        assert len([c for _,c in commands if c=="end_run"])==ranks
        assert len([c for _,c in commands if c=="idle"])==ranks
        assert all(not p.exists() for p in inputs+outputs)
        assert session._wait_state is None
        assert any(e.get("retention")=="DRAINING_CURRENT_RPC" for e in session.history)
    finally:
        released.set();session._drain_done.wait(5)
        session.processes=[];session.close()


def test_release_before_vae_honors_h3_retention(tmp_path,monkeypatch):
    import powershard.runtime as runtime
    kept=Session(None,DistributedConfig(weight_placement="cpu",release_after_sampling=False),tmp_path,tmp_path)
    discarded=Session(None,DistributedConfig(weight_placement="cpu",release_after_sampling=True),tmp_path,tmp_path)
    calls=[];kept.idle=lambda:calls.append("kept_idle");kept.close=lambda **kw:calls.append("kept_close")
    discarded.close=lambda **kw:calls.append("discarded_close")
    monkeypatch.setattr(runtime,"_SESSIONS",[kept,discarded])
    release_all(preserve_h3_cpu=True)
    assert calls==["kept_idle","discarded_close"]
    calls.clear();release_all()
    assert calls==["kept_close","discarded_close"]


def test_real_cpu_worker_survives_comfy_interrupt_and_next_rpc(h3_factory,tmp_path):
    import torch
    import comfy.model_management as mm
    from safetensors.torch import save_file
    from test_native_pipeline import CPUContractSession
    net=h3_factory()
    checkpoint=tmp_path/"tiny.safetensors";save_file(net.state_dict(),str(checkpoint))
    comfy_path=Path(mm.__file__).resolve().parents[1]
    session=CPUContractSession(str(checkpoint),DistributedConfig(weight_placement="cpu",release_after_sampling=False),
                               comfy_path,tmp_path/"reports",patch=H3PatchConfig(enabled=True))
    gate=tmp_path/"gate";gate.mkdir();session._test_pause_dir=gate
    text=torch.ones(1,5,24)
    def cancel():
        if (gate/"ready").exists():raise mm.InterruptProcessingException()
    try:
        with pytest.raises(mm.InterruptProcessingException):session.call("preprocess_text",(text,),{},cancel)
        assert session.draining and session.running
        pid=session.processes[0].pid
        session.finish_sampling()
        (gate/"continue").touch()
        assert session._drain_done.wait(10)
        assert session.running and session.idle_on_cpu and session.processes[0].pid==pid
        result=session.call("preprocess_text",(text,),{})
        assert result.shape==(1,5,32) and torch.isfinite(result).all()
        assert session.processes[0].pid==pid
        assert len([e for e in session.history if e.get("status")=="DISCARDED_AFTER_INTERRUPT"])==1
    finally:
        (gate/"continue").touch();session._drain_done.wait(10);session.close()
