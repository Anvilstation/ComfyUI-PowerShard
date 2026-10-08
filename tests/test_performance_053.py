"""Real CPU tensors/native H3 plus mocked DTensor ownership, NOT CUDA acceptance."""
import gc
import json
import types
import weakref
from types import SimpleNamespace
import pytest
import torch
from powershard.config import DistributedConfig, shard_bounds
from powershard.phase_cache import LocalShardState, ResidentBackend, ShardBindings


def emulate_local_shards(root, rank=0, world=1):
    """No process group: Parameter storage is a CPU mock of a local DTensor."""
    for param in root.parameters():
        param.placements = (SimpleNamespace(dim=0),)
        a, b = shard_bounds(param.shape[0], rank, world)
        param.to_local = types.MethodType(lambda self, a=a, b=b: self.detach()[a:b], param)


def tiny_root(rank=0, world=5):
    from powershard.operations import Linear
    root = torch.nn.Module()
    root.layer = Linear(4, 13, bias=True, dtype=torch.float16).requires_grad_(False)
    root.layer.weight.copy_(torch.arange(52).reshape(13,4)*.001)
    root.layer.bias.zero_()
    root.layer._ps_safe, root.layer._ps_fp32 = True, False
    root.layer._ps_weight_bounds = (2., 3.)
    root.register_buffer("phase_buffer", torch.tensor([2., 7.]))
    emulate_local_shards(root, rank, world)
    return root


@pytest.mark.parametrize("rank", range(5))
def test_cpu_cache_copies_only_local_rows_and_restores_without_checkpoint(rank, monkeypatch):
    import powershard.fsdp_backend as fsdp
    import safetensors
    source = tiny_root(rank)
    with torch.no_grad():
        source.layer.weight.copy_(torch.arange(52).reshape(13,4))
        source.layer.bias.copy_(torch.arange(13))
    cache = ShardBindings(source).capture()
    a, b = shard_bounds(13, rank, 5)
    assert cache.parameter_bytes == (b-a)*5*2
    assert all(t.device.type == "cpu" for t in cache.parameters.values())
    assert cache.parameters["layer.weight"].data_ptr() != source.layer.weight.to_local().data_ptr()
    source.layer.weight.zero_()
    target = tiny_root(rank)
    monkeypatch.setattr(fsdp, "DTensor", torch.nn.Parameter)
    monkeypatch.setattr(safetensors, "safe_open", lambda *a, **kw: pytest.fail("checkpoint weights read on resume"))
    evidence = fsdp.load_local(target, SimpleNamespace(path="missing.safetensors"), torch.device("cpu"),
                               rank, 5, local_state=cache)
    torch.testing.assert_close(target.layer.weight.to_local(), torch.arange(52).reshape(13,4)[a:b].half(), rtol=0, atol=0)
    torch.testing.assert_close(target.phase_buffer, torch.tensor([2.,7.]), rtol=0, atol=0)
    assert target.layer._ps_weight_bounds == (2.,3.)
    assert sum(s["local_bytes"] for s in evidence) == cache.parameter_bytes
    assert {s["source"] for s in evidence} == {"RAM_LOCAL_SHARD_CACHE"}


@pytest.mark.parametrize("corruption", ["names", "dtype", "shape", "bounds"])
def test_cache_restore_refuses_incompatible_state(corruption, monkeypatch):
    import powershard.fsdp_backend as fsdp
    cache = ShardBindings(tiny_root()).capture()
    if corruption == "names":
        cache.parameters["wrong"] = cache.parameters.pop("layer.weight")
    elif corruption == "dtype":
        cache.parameters["layer.weight"] = cache.parameters["layer.weight"].float()
    elif corruption == "shape":
        cache.parameters["layer.weight"] = cache.parameters["layer.weight"][:1]
    else:
        cache.weight_bounds.clear()
    monkeypatch.setattr(fsdp, "DTensor", torch.nn.Parameter)
    with pytest.raises(RuntimeError, match="Phase cache"):
        fsdp.load_local(tiny_root(), None, torch.device("cpu"), 0, 5, local_state=cache)


def test_non_sharded_model_cannot_be_parked():
    root = torch.nn.Linear(4, 13)
    with pytest.raises(RuntimeError, match="resharded DTensor"):
        ShardBindings(root).capture()


@pytest.mark.parametrize("placement", ["gpu", "ats"])
def test_owner_drops_live_graph_and_managed_pool_and_resumes_exactly(placement, monkeypatch):
    import powershard.fsdp_backend as fsdp
    monkeypatch.setattr(fsdp, "assert_sharded", lambda *a: None)
    monkeypatch.setattr(fsdp, "DTensor", torch.nn.Parameter)
    class Backend:
        def __init__(self, cache=None):
            self.root = tiny_root()
            if cache:
                fsdp.load_local(self.root, None, torch.device("cpu"), 0, 5, local_state=cache)
            self.state_bindings = ShardBindings(self.root)
            self.config = DistributedConfig(weight_placement=placement)
            self.units = []
            self.shard_bytes = sum(p.to_local().numel()*p.element_size() for p in self.root.parameters())
            self.managed_pool = types.SimpleNamespace()  # lifetime tested, NOT real Unified Memory
            # Deliberate module/hook cycle: GC at parking must collect it.
            self.root._owner = self
        def call(self, command, args, kwargs):
            return args[0] @ self.root.layer.weight.to_local().T, {}
    backend = Backend()
    old = weakref.ref(backend)
    owner = ResidentBackend(backend, Backend, torch.device("cpu"))
    backend = None
    x = torch.ones(2,4).half()
    reference, _ = owner.call("forward", (x,), {})
    report = owner.idle()
    assert old() is None and owner._backend is None and owner._cached is not None
    assert report["cpu_shard_bytes"] == 30 and report["checkpoint_weight_reads"] == 0
    assert not hasattr(owner, "root")  # introspection/end_run may not trigger a reload
    assert owner.end_run() is None
    assert owner.idle()["already_idle"]
    actual, metrics = owner.call("forward", (x,), {})
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert metrics["phase_cache"]["resumed_from_ram"] and owner._cached is None
    assert not owner.call("forward", (x,), {})[1]["phase_cache"]["resumed_from_ram"]
    owner.idle()
    gc.collect()


def test_failed_resume_preserves_ram_cache(monkeypatch):
    def broken(cache):
        raise RuntimeError("injected reconstruction failure")
    owner = ResidentBackend(None, broken, torch.device("cpu"))
    cached = LocalShardState({"weight": torch.ones(3,4)}, {}, {})
    owner._cached = cached
    with pytest.raises(RuntimeError, match="reconstruction"):
        owner.call("forward", (), {})
    assert owner._cached is cached and owner._backend is None


def test_idle_status_uses_parking_measurement_not_previous_vram(tmp_path, monkeypatch):
    import powershard.runtime as runtime
    from powershard.web_api import runtime_status
    session = runtime.Session(None,DistributedConfig(weight_placement="ats",release_after_sampling=False),tmp_path,tmp_path)
    session.processes = [SimpleNamespace(poll=lambda:None)]
    session.idle_on_cpu = True
    session.history = [dict(ranks=[dict(metrics=dict(memory=dict(allocated=1000),weight_memory=dict(gpu_shard_bytes=900)))]),
                       dict(ranks=[dict(phase_offload=dict(after=dict(allocated=0),cpu_shard_bytes=900))])]
    monkeypatch.setattr(runtime,"_SESSIONS",[session])
    status=runtime_status()[0]
    assert status["placement"]=="ats" and status["idle_placement"]=="cpu"
    assert status["ranks"][0]["memory"]["allocated"]==0
    assert status["ranks"][0]["weights"]["cpu_shard_bytes"]==900
    assert status["ranks"][0]["weights"]["managed_shard_bytes"]==0
    session.processes=[]


def test_parked_zero_vram_is_not_replaced_by_active_memory_budget(h3_factory,tmp_path):
    from powershard.comfy_adapter import PowerShardPatcher
    from powershard.qwen_adapter import QwenPatcher
    for cls in (PowerShardPatcher,QwenPatcher):
        mock=SimpleNamespace(session=SimpleNamespace(running=True,idle_on_cpu=True,last_memory=0,
                             config=DistributedConfig(weight_placement="ats")),
                             powershard_memory_plan={"host_gpu_parameter_budget_bytes":2**33})
        assert cls.loaded_size(mock)==0


@pytest.mark.parametrize("placement", ["gpu", "cpu", "ats"])
@pytest.mark.parametrize("action", ["deactivate", "finish_sampling", "release"])
def test_retention_parks_every_active_placement_in_ram(tmp_path, monkeypatch, placement, action):
    import powershard.runtime as runtime
    session = runtime.Session(None, DistributedConfig(weight_placement=placement, release_after_sampling=False), tmp_path, tmp_path)
    session.processes = [SimpleNamespace(poll=lambda: None)]
    calls = []
    monkeypatch.setattr(session, "idle", lambda: calls.append("idle"))
    monkeypatch.setattr(session, "close", lambda **kw: calls.append("close"))
    monkeypatch.setattr(session, "control", lambda command: calls.append(command))
    monkeypatch.setattr(runtime, "_SESSIONS", [session])
    if action == "release":
        runtime.release_all(preserve_h3_cpu=True)
    else:
        getattr(session, action)()
    assert calls == (["end_run", "idle"] if action == "finish_sampling" else ["idle"])
    session.processes = []


@pytest.mark.parametrize("mode", ["off", "manual"])
def test_fixed_mlp_mode_never_samples_live_allocator(monkeypatch, mode):
    from test_mlp_policy import mlp
    import powershard.memory_policy as policy
    from powershard.patch_config import H3PatchConfig
    monkeypatch.setattr(policy, "mlp_budget", lambda *a: pytest.fail("live allocator called for fixed MLP mode"))
    block = mlp()
    block._ps_mlp_policy = H3PatchConfig(True, True, False, 4, mode)
    block._ps_mlp_chunk = 4
    with torch.no_grad():
        result = block(torch.randn(11,16))
    assert torch.isfinite(result).all()
    assert block._ps_mlp_report["budget_source"].startswith("RPC boundary")


def test_progress_is_throttled_but_first_mlp_and_final_status_persist(tmp_path, monkeypatch):
    import powershard.telemetry as telemetry
    tick = [1.]
    monkeypatch.setattr(telemetry.time, "perf_counter", lambda: tick[0])
    progress = telemetry.RequestProgress(tmp_path/"progress.json")
    progress.begin(1,"forward")
    progress("mlp_plan", {"module": "first", "effective_tokens": 9307})
    for i in range(50):
        progress("mlp_plan", {"module": str(i), "effective_tokens": 9307})
    assert progress.report()["atomic_writes"] == 2
    assert json.loads(progress.path.read_text())["mlp_plan"]["module"] == "first"
    tick[0] += .6
    progress("mlp_plan", {"module": "after_interval"})
    progress("status", "COMPLETE")
    saved = json.loads(progress.path.read_text())
    assert saved["status"] == "COMPLETE" and saved["mlp_plan"]["module"] == "after_interval"
    assert progress.report()["atomic_writes"] == 4


def test_baseline_wire_and_prefetch_controls_do_change_config():
    from powershard.nodes import PowerShardConfigTuning
    base = DistributedConfig()
    assert base.sequence_comm_dtype == "fp32"
    changed = PowerShardConfigTuning().tune(base, prefetch_blocks=2, sequence_comm_dtype="fp16", prefetch_policy="manual")[0]
    assert changed.sequence_comm_dtype == "fp16" and changed.memory_policy == "manual" and changed.prefetch_blocks == 2
    from powershard.memory_policy import plan_forward
    auto = plan_forward(1,2,3,4,5,2,"auto")
    manual = plan_forward(1,2,3,4,5,2,"manual")
    assert auto["effective_prefetch"] == 0 and auto["prefetch_reduced"]
    assert "insufficient" in auto["prefetch_reason"]
    assert manual["effective_prefetch"] == 2 and not manual["prefetch_reduced"]


def test_schema6_tuning_appends_new_widgets_without_reordering():
    from powershard.workflow_migration import migrate_ui, migrate_api
    old = [2.,1,"auto",True,False,False]
    graph = dict(nodes=[dict(id=1,type="PowerShardConfigTuning",widgets_values=old)],links=[])
    new, _ = migrate_ui(graph)
    assert new["nodes"][0]["widgets_values"] == old + ["fp32","auto"]
    assert migrate_ui(new) == (new, [])
    graph = {"1": dict(class_type="PowerShardConfig", inputs=dict(gpu_ids="0,1,2,3,4", sequence_comm_dtype="fp16"))}
    new, _ = migrate_api(graph)
    tuning = next(n for n in new.values() if n["class_type"] == "PowerShardConfigTuning")
    assert tuning["inputs"]["sequence_comm_dtype"] == "fp16"


def test_native_h3_cpu_cache_roundtrip_with_canonical_names_under_gates(h3_factory, monkeypatch):
    from powershard.fsdp_backend import Entrypoint, load_local
    import powershard.fsdp_backend as fsdp
    from powershard.fp16_safe import apply_fp16_safe
    from powershard.operations import Linear, checkpoint_row_bounds, matmul_constants
    from powershard.patch_config import H3PatchConfig
    from powershard.spectrum import SpectrumBlockGate, SpectrumEngine, install_spectrum
    from powershard.spectrum_config import SpectrumConfig
    def build():
        net = h3_factory()
        tracker = apply_fp16_safe(net, H3PatchConfig(enabled=True))
        for module in net.modules():
            if isinstance(module, Linear) and not module._ps_fp32 and module.weight.dtype == torch.float16:
                module._ps_weight_bounds = matmul_constants(*checkpoint_row_bounds(module.weight), module.in_features)
        root = Entrypoint(net)
        emulate_local_shards(root)
        return root, tracker
    root, tracker = build()
    g = torch.Generator().manual_seed(2045)
    text = torch.randn(1,7,24,generator=g)*1e5
    args = ([torch.randn(1,24,2,4,4,generator=g),torch.randn(1,32,2,5,generator=g)],torch.tensor([700.]))
    with torch.no_grad():
        tracker.begin("cpu")
        context = root("preprocess_text",(text,),{})
        tracker.finish(context)
        tracker.begin("cpu")
        before = root("forward",(*args,context),{})
        tracker.finish(before)
    bindings = ShardBindings(root)
    install_spectrum(root.network, SpectrumEngine(SpectrumConfig(enabled=True),"cpu"))
    assert isinstance(root.network.blocks[0],SpectrumBlockGate)
    assert any(".block." in name for name, _ in root.named_parameters())
    cache = bindings.capture()
    assert not any(".block." in name for name in cache.parameters)
    monkeypatch.setattr(fsdp,"DTensor",torch.nn.Parameter)
    target, tracker = build()
    load_local(target,None,torch.device("cpu"),0,1,local_state=cache)
    with torch.no_grad():
        tracker.begin("cpu")
        context_new = target("preprocess_text",(text,),{})
        tracker.finish(context_new)
        tracker.begin("cpu")
        after = target("forward",(*args,context_new),{})
        tracker.finish(after)
    torch.testing.assert_close(context_new,context,rtol=0,atol=0)
    for x,y in zip(after,before):
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_h3_qk_metadata_resume_reads_no_checkpoint_norm_weights(h3_factory,monkeypatch):
    from powershard.attention import configure_safe_qk
    import safetensors
    net=h3_factory()
    scales={name:(8.,4.) for name,module in net.named_modules() if hasattr(module,"_ps_attention")}
    net.to_empty(device="meta")
    monkeypatch.setattr(safetensors,"safe_open",lambda *a,**kw:pytest.fail("QK norm weights read on resume"))
    configure_safe_qk(net,None,scales)
    assert all(net.get_submodule(name)._ps_qk_scales==(8.,4.) for name in scales)


def test_phase_history_is_available_without_closing_ram_owner(tmp_path):
    from powershard.runtime import Session
    session=Session(None,DistributedConfig(),tmp_path,tmp_path/"reports")
    session.path=tmp_path/"powershard-session"
    session.history=[dict(sequence=2,command="forward",ranks=[dict(metrics=dict(forward_s=1.))])]
    session.flush_history()
    assert json.loads((tmp_path/"reports/powershard-session.json").read_text())==session.history
    assert session.path==tmp_path/"powershard-session"
    session.path=None
