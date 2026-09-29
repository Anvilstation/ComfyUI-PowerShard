"""CPU контракты RAM-профиля. Эти тесты НЕ доказывают CUDA/NCCL/offload."""
import copy
import json
import types
from pathlib import Path
import pytest
import torch
from powershard.config import DistributedConfig
from powershard.attention_contract import AttentionOptions as O, sdpa_attention_bounded, sdpa_tile_plan, math_attention
from powershard.attention_policy import AttentionDispatcher
from powershard.memory_policy import apply_workspace_policy, apply_linear_policy
from test_attention_contract import tensors, reference


@pytest.mark.parametrize('count', [1,2,3,4,5,6,9])
def test_profile_normalizes_without_gpu_gate(count):
    cfg=DistributedConfig(tuple(map(str,range(count))),memory_profile='ram_min',prefetch_blocks=2)
    assert len(cfg.gpu_ids)==count and cfg.cpu_offload and cfg.prefetch_blocks==0 and cfg.memory_policy=='auto'
    assert DistributedConfig(**cfg.to_dict())==cfg
    old=DistributedConfig();assert not old.cpu_offload and not old.min_vram
    plan=apply_workspace_policy(dict(mlp_budget_bytes=2**40,effective_prefetch=2),cfg)
    assert plan['mlp_budget_bytes']==256*2**20 and not plan['allow_prepared_weights']
    assert plan['mlp_mode_override']=='auto' and plan['effective_prefetch']==0
    assert cfg.memory_settings()['conditioning_cache_device']=='cpu'


@pytest.mark.parametrize('kwargs',[dict(workspace_mib=0),dict(stage_cache_mib=-1),dict(memory_profile='unlimited')])
def test_invalid_profile_arguments(kwargs):
    with pytest.raises(ValueError):DistributedConfig(**kwargs)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
@pytest.mark.parametrize('kind',['plain','causal_upper','causal_bottom','bool','bias','window','alibi','combined'])
def test_sdpa_tiles_preserve_dense_semantics(dtype,kind):
    q,k,v=tensors(lq=9,lk=13,hq=6,hk=2,dv=5,dtype=dtype)
    args=dict(softmax_scale=.173)
    if kind.startswith('causal') or kind=='combined':args.update(causal=True,causal_alignment='bottom_right' if kind.endswith('bottom') else 'upper_left')
    if kind in ('bool','combined'):args['mask']=torch.arange(13)[None,:]!=3
    if kind=='bias':args['mask']=torch.linspace(-.3,.8,13).reshape(1,1,1,13).expand(2,6,9,13)
    if kind in ('window','combined'):args['window_size']=(2,1)
    if kind=='alibi':args['alibi_slopes']=torch.linspace(.01,.1,6)
    opts=O(**args)
    actual=sdpa_attention_bounded(q,k,v,opts,workspace_bytes=2*2*13*16,query_chunk=2)
    expected=reference(q,k,v,opts)
    assert actual.shape==(2,9,6,5) and actual.dtype==dtype and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(),expected,atol=.0015 if dtype==torch.float16 else 2e-6,rtol=.003 if dtype==torch.float16 else 2e-5)


def test_tiles_batch_alibi_all_masked_and_reverse_rectangular():
    q,k,v=tensors(lq=9,lk=4,hq=6,hk=1)
    mask=torch.ones(2,6,9,4,dtype=torch.bool);mask[:,:,6]=False
    opts=O(causal=True,causal_alignment='bottom_right',mask=mask,alibi_slopes=torch.rand(2,6)*.1)
    actual=sdpa_attention_bounded(q,k,v,opts,256,2)
    expected=math_attention(q,k,v,opts,2,2)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    assert torch.count_nonzero(actual[:,:5])==0 and torch.count_nonzero(actual[:,6])==0


def test_sdpa_budget_applies_to_real_calls_and_no_default_scale(monkeypatch):
    q,k,v=tensors(lq=17,lk=11,hq=6,hk=2)
    original=torch.nn.functional.scaled_dot_product_attention;seen=[]
    def wrapped(q,k,v,**kw):
        seen.append((q.shape,k.shape,kw.copy()))
        return original(q,k,v,**kw)
    monkeypatch.setattr(torch.nn.functional,'scaled_dot_product_attention',wrapped)
    got=sdpa_attention_bounded(q,k,v,O(softmax_scale=0.,dropout_p=.2),704,4)
    assert len(seen)>1 and torch.isfinite(got).all()
    assert all(a[0]*a[1]*a[2]*b[2]*16<=704 and opts['scale']==0. and opts['dropout_p']==.2 for a,b,opts in seen)
    assert sdpa_tile_plan(q,k,1,128)['minimum_tile_exceeds_budget']


def test_profile_fallback_counts_and_fatal_errors(monkeypatch):
    cfg=DistributedConfig(memory_profile='ram_min',attention_backend='sdpa',workspace_mib=1)
    dispatch=AttentionDispatcher(cfg);q,k,v=tensors()
    out=dispatch(q,k,v,O(),group='dit')
    assert dispatch.report()['call_counts']=={'dit:sdpa_tiled':1}
    with pytest.warns(UserWarning):triple=dispatch(q,k,v,O(return_attn_probs=True),group='refiner')
    assert len(triple)==3 and dispatch.report()['call_counts']['refiner:math']==1
    def oom(*a,**kw):raise torch.OutOfMemoryError('injected; must terminate session')
    monkeypatch.setattr(torch.nn.functional,'scaled_dot_product_attention',oom)
    with pytest.raises(torch.OutOfMemoryError):dispatch(q,k,v,O())


@pytest.mark.parametrize('fp32',[False,True])
def test_dense_projection_weight_tiling_keeps_safe_range(fp32):
    from powershard.operations import Linear
    from powershard.fp16_safe import FiniteTracker
    torch.manual_seed(71)
    layer=Linear(32,29,bias=True,dtype=torch.float16).requires_grad_(False)
    layer.weight.normal_(std=.03);layer.bias.normal_(std=.01)
    layer._ps_safe=True;layer._ps_fp32=fp32
    layer._ps_tracker=FiniteTracker();layer._ps_finite_slot=layer._ps_tracker.register('linear')
    x=torch.randn(2,11,32)*1e6
    with torch.no_grad():
        ref=layer(x)
        apply_linear_policy(layer,DistributedConfig(memory_profile='ram_min',dequant_rows=7))
        out=layer(x)
    assert out.dtype==torch.float32 and torch.isfinite(out).all()
    torch.testing.assert_close(out,ref,atol=256,rtol=.003)


@pytest.mark.parametrize('quant',[False,True])
def test_ram_mlp_overrides_off_without_prepared_full_weights(quant,monkeypatch):
    from test_mlp_policy import mlp
    from powershard.patch_config import H3PatchConfig
    from powershard.operations import prepared_linears
    m=mlp(quant);m._ps_mlp_policy=H3PatchConfig(mlp_chunk_mode='off');m._ps_mlp_chunk=512
    x=torch.randn(1,11,16)*1e5
    with torch.no_grad():ref=m(x)
    cfg=DistributedConfig(memory_profile='ram_min',dequant_rows=7)
    apply_linear_policy(m,cfg)
    m._ps_memory_context=apply_workspace_policy(dict(mlp_budget_bytes=5000),cfg)
    with torch.no_grad():out=m(x)
    assert m._ps_mlp_report['mode']=='auto' and m._ps_mlp_report['effective_tokens']<11
    torch.testing.assert_close(out,ref,atol=16,rtol=.004)
    assert not any(hasattr(v,'_ps_prepared') for v in m.modules())
    if quant:assert m.fc1.weight.dtype==torch.int8 and m.fc1.convrot


def test_bounded_stage_cache_owns_storage_and_evicts(tmp_path):
    from powershard.wire import write_payload,read_payload,StagedTensor
    stage=tmp_path/'stage';step=tmp_path/'step'
    write_payload(stage,[torch.full((16,),float(i)) for i in range(4)])
    cache={}
    def read(index,limit=128):
        write_payload(step,StagedTensor('t'+str(index)))
        return read_payload(step,'cpu',stage,cache,cache_device='cpu',cache_limit_bytes=limit)
    for i in (0,1,0,2,3,0):
        value=read(i);value.fill_(-1)
        entries={k:v for k,v in cache.items() if isinstance(v,torch.Tensor)}
        assert sum(v.untyped_storage().nbytes() for v in entries.values())<=128
        assert all(torch.all(v>=0) for v in entries.values())
    assert 't1' not in cache
    assert torch.all(read(0)==0)
    read(0,0);assert not any(isinstance(v,torch.Tensor) for v in cache.values())


def test_nested_wrap_plan_native_calls_and_all_groups_reshard(h3_factory,monkeypatch):
    import powershard.fsdp_backend as backend
    from powershard.fp16_safe import apply_fp16_safe
    from powershard.patch_config import H3PatchConfig
    net=h3_factory();root=backend.Entrypoint(net);records=[];calls=[]
    def fake_shard(unit,**kwargs):
        # Проверяем выбор/порядок hooks; НЕ заменяет настоящий FSDP/CUDA тест.
        records.append((unit,kwargs));unit.set_modules_to_forward_prefetch=lambda modules:None
        unit.register_forward_pre_hook(lambda module,args:calls.append(module))
    monkeypatch.setattr(backend,'fully_shard',fake_shard)
    units=backend.wrap_fsdp(root,None,DistributedConfig(memory_profile='ram_min'))
    block=net.blocks[0]
    assert units.index(block.attn)<units.index(block) and units.index(block.mlp)<units.index(block)
    assert all(kw['reshard_after_forward'] is True and type(kw['offload_policy']).__name__=='CPUOffloadPolicy' for _,kw in records)
    apply_fp16_safe(net,H3PatchConfig(enabled=True))
    with torch.no_grad():
        ctx=root('preprocess_text',(torch.ones(1,7,24),),{})
        root('forward',([torch.zeros(1,24,2,4,4),torch.zeros(1,32,2,5)],torch.tensor([700.]),ctx),{})
    assert block.attn in calls and block.mlp in calls and net.token_refiner.blocks[0].attn in calls


def test_native_h3_ram_profile_preprocess_forward(h3_factory):
    from powershard.fp16_safe import apply_fp16_safe
    from powershard.patch_config import H3PatchConfig
    from powershard.attention import install_attention
    net=h3_factory();ref=copy.deepcopy(net)
    cfg=DistributedConfig(memory_profile='ram_min',attention_backend='sdpa',dequant_rows=7,workspace_mib=1)
    dispatch=AttentionDispatcher(cfg);install_attention(net,cfg,dispatch)
    for module in (net,ref):apply_fp16_safe(module,H3PatchConfig(enabled=True,debug_finite=True))
    apply_linear_policy(net,cfg)
    plan=apply_workspace_policy(dict(mlp_budget_bytes=4096),cfg)
    for module in net.modules():
        if hasattr(module,'_ps_mlp_chunk'):module._ps_memory_context=plan
    args=[torch.ones(1,24,2,4,4),torch.ones(1,32,2,5)]
    with torch.no_grad():
        text=torch.full((1,7,24),1e5)
        ctx=net.preprocess_text_embeds(text);ctx_ref=ref.preprocess_text_embeds(text)
        for _ in range(3):
            actual=net(args,torch.tensor([700.]),ctx);expected=ref(args,torch.tensor([700.]),ctx_ref)
            for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=.003,rtol=.004)
    counts=dispatch.report()['call_counts']
    assert counts['dit:sdpa_tiled']>0 and counts['token_refiner:sdpa_tiled']>0


def test_ui_schema_tooltips_and_appended_widget_fields(monkeypatch):
    from powershard.nodes import NODE_CLASS_MAPPINGS
    from powershard.ui_schema import schema
    monkeypatch.setitem(__import__('sys').modules,'folder_paths',types.SimpleNamespace(get_filename_list=lambda _:['local.safetensors']))
    for name,cls in NODE_CLASS_MAPPINGS.items():
        assert cls.DESCRIPTION and schema()['nodes'][name]['title']
        for fields in cls.INPUT_TYPES().values():
            assert all(len(spec)>1 and spec[1]['tooltip'] for spec in fields.values())
    cls=NODE_CLASS_MAPPINGS['PowerShardConfig'];inputs=cls.INPUT_TYPES()
    keys=list(inputs['required'])+list(inputs['optional'])
    assert keys[:17]==['gpu_ids','backend','precision','reserve_gib','timeout_s','allow_unverified','release_after_sampling','cpu_offload','pin_memory','prefetch_blocks','numa_policy','attention_backend','allow_fallback','memory_policy','weight_placement','sequence_mode','sequence_comm_dtype']
    assert keys[17:]==['memory_profile','workspace_mib','stage_cache_mib']
    old=cls().create('5,2,0','fsdp2','fp16',2.,600,False,True)[0]
    new=cls().create('all','fsdp2','fp16',2.,600,False,True,memory_profile='ram_min')[0]
    assert not old.min_vram and new.cpu_offload and new.gpu_ids==('all',)


def test_frontend_settings_and_old_workflows(tmp_path):
    import subprocess,shutil
    from powershard.nodes import PowerShardConfig
    from powershard.ui_schema import schema
    node=shutil.which('node')
    if node is None:pytest.skip('NOT_RUN: Node.js отсутствует; native UI/browser требует отдельного запуска')
    root=Path(__file__).resolve().parents[1]
    meta=tmp_path/'ui.json';meta.write_text(json.dumps(dict(schema=schema(),input=PowerShardConfig.INPUT_TYPES())))
    completed=subprocess.run([node,str(root/'tests/ui_settings_contract.cjs'),str(root/'web/powershard.js'),str(meta)],text=True,capture_output=True,timeout=30)
    assert completed.returncode==0,completed.stdout+completed.stderr


def test_ram_workflows_preserve_widgets_and_av_pipeline():
    from powershard.nodes import PowerShardConfig
    inputs=PowerShardConfig.INPUT_TYPES();keys=list(inputs['required'])+list(inputs['optional'])
    root=Path(__file__).resolve().parents[1]
    for path in (root/'workflows').glob('*ram_min*.api.json'):
        graph=json.loads(path.read_text());ui=json.loads(Path(str(path).replace('.api.','.ui.')).read_text())
        for row in ui['nodes']:
            if row['type']=='PowerShardConfig':
                actual=graph[str(row['id'])]['inputs']
                assert row['widgets_values']==[actual[key] for key in keys]
                assert actual['memory_profile']=='ram_min'
        assert graph['20']['class_type']=='SaveImage' and graph['15']['class_type']=='VAEDecodeAudio'
        assert graph['1']['inputs']['cpu_offload'] and graph['3']['inputs']['idle_policy']=='cpu_shards'


def test_warm_allocator_budget_not_only_driver_free(monkeypatch):
    from powershard.memory_policy import allocator_budget
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(100,2000))
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:1400)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:400)
    monkeypatch.setattr(torch.cuda,'memory_stats',lambda device:{'inactive_split_bytes.all.current':200})
    monkeypatch.setattr(torch.cuda.memory,'get_allocator_backend',lambda:'native')
    got=allocator_budget(torch.device('cuda:0'))
    assert got['driver_free_bytes']==100 and got['planning_available_bytes']==900
    monkeypatch.setattr(torch.cuda.memory,'get_allocator_backend',lambda:'cudaMallocAsync')
    assert allocator_budget(torch.device('cuda:0'))['planning_available_bytes']==100


def test_memory_benchmark_compares_effective_profiles_and_rejects_overrides(tmp_path):
    import subprocess,sys
    root=Path(__file__).resolve().parents[1]
    source=root/'workflows/fl2va_ram_min_int8.api.json'
    command=[sys.executable,str(root/'scripts/benchmark_cases.py'),str(source),'--output-dir',str(tmp_path)]
    proc=subprocess.run(command+['--axis','memory'],capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    for case in manifest['cases']:
        graph=json.loads(Path(case['workflow']).read_text())
        cfg=DistributedConfig(**graph['1']['inputs'])
        assert cfg.memory_profile==case['value'] and cfg.cpu_offload
    for axis in ('offload','mlp'):
        proc=subprocess.run(command+['--axis',axis],capture_output=True,text=True,timeout=30)
        assert proc.returncode!=0 and 'ram_min' in proc.stderr
