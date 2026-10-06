import copy
import json
from pathlib import Path
import pytest
from powershard.workflow_migration import migrate_api, migrate_ui

ROOT=Path(__file__).resolve().parents[1]


def assert_acyclic_api(graph):
    active, done=set(),set()
    def visit(key):
        assert key not in active, f"cycle through {key}"
        if key in done:return
        active.add(key)
        for value in graph[key]['inputs'].values():
            if isinstance(value,list) and len(value)==2 and str(value[0]) in graph:
                visit(str(value[0]))
        active.remove(key);done.add(key)
    for key in graph:visit(key)


def test_old_api_values_and_connections():
    original={'1':dict(class_type='PowerShardConfig',inputs=dict(gpu_ids='5,2,0',backend='fsdp2',precision='int8_fp16',
                timeout_s=10,reserve_gib=1.5,cpu_offload=True,release_after_sampling=False,prefetch_blocks=1,numa_policy='auto',pin_memory=False)),
        '2':dict(class_type='PowerShardH3Loader',inputs=dict(config=['1',0],checkpoint='model.safetensors')),
        '3':dict(class_type='PowerShardH3FP16Patcher',inputs=dict(model=['2',0],enabled=True,fp16_safe=True,debug_finite=True,
                mlp_chunk_mode='manual',mlp_chunk_tokens=1024)),
        '4':dict(class_type='BasicGuider',inputs=dict(model=['3',0])),
        '5':dict(class_type='PowerShardH3QwenLoader',inputs=dict(config=['1',0],checkpoint='qwen.safetensors',precision='int8_fp16',
                idle_policy='release',cache_mib=128,mlp_chunk_mode='auto',mlp_chunk_tokens=8192)),
        '6':dict(class_type='MiniMaxH3ImageToVideo',inputs=dict(clip=['5',0]))}
    before=copy.deepcopy(original);graph,changes=migrate_api(original)
    assert original==before and changes
    assert graph['1']['inputs']['weight_placement']=='cpu' and graph['1']['inputs']['gpu_ids']=='5,2,0'
    tuning=graph[graph['2']['inputs']['config'][0]]
    assert tuning['class_type']=='PowerShardConfigTuning'
    assert graph['2']['inputs']['keep_in_memory'] and not tuning['inputs']['pin_memory']
    mlp=graph[graph['4']['inputs']['model'][0]]
    assert mlp['class_type']=='PowerShardH3MLP' and mlp['inputs']['model']==['3',0]
    assert mlp['inputs']['chunk_tokens']==1024
    qmlp=graph[graph['6']['inputs']['clip'][0]]
    assert qmlp['class_type']=='PowerShardQwenMLP' and qmlp['inputs']['chunk_tokens']==8192
    assert_acyclic_api(graph)
    assert migrate_api(graph)==(graph,[])


@pytest.mark.parametrize('layout',[14,18])
def test_old_ui_positional_values(layout):
    data=json.loads((ROOT/'workflows/fl2va_fp16.ui.json').read_text())
    config=next(n for n in data['nodes'] if n['type']=='PowerShardConfig')
    patcher=next(n for n in data['nodes'] if n['type']=='PowerShardH3FP16Patcher')
    config['widgets_values']=['5,2,0','fsdp2','fp16',1.5,20,True,False,True,False,1,'auto']
    config['widgets_values']+=['sdpa',False,'auto'] if layout==14 else ['cpu','sdpa',False,'auto','ulysses','fp16',False]
    patcher['widgets_values']=[True,True,True,1024,'manual']
    before=copy.deepcopy(data);result,changes=migrate_ui(data)
    assert data==before and changes
    config=next(n for n in result['nodes'] if n['id']==config['id'])
    patcher=next(n for n in result['nodes'] if n['id']==patcher['id'])
    assert config['widgets_values']==['5,2,0','cpu','fp16','sdpa','token' if layout==14 else 'ulysses']
    assert patcher['widgets_values']==[True,True]
    ids={n['id']:n for n in result['nodes']}
    for link,source,slot,target,input_slot,dtype in result['links']:
        assert ids[target]['inputs'][input_slot]['link']==link
        assert link in ids[source]['outputs'][slot]['links']
    assert migrate_ui(result)==(result,[])


def test_reject_ambiguous_ui_layout():
    data=dict(nodes=[dict(id=1,type='PowerShardConfig',widgets_values=[0]*12)],links=[])
    with pytest.raises(ValueError,match='layout'):migrate_ui(data)


def test_linked_legacy_boolean_policy_is_never_silently_dropped():
    original={'1':dict(class_type='PowerShardConfig',inputs=dict(cpu_offload=['2',0]))}
    with pytest.raises(ValueError,match='boolean'):migrate_api(original)


def test_all_shipped_workflows_are_current_and_acyclic():
    from powershard.nodes import NODE_CLASS_MAPPINGS
    for path in (ROOT/'workflows').glob('*.api.json'):
        graph=json.loads(path.read_text());assert_acyclic_api(graph)
        assert migrate_api(graph)==(graph,[])
        for node in graph.values():
            cls=NODE_CLASS_MAPPINGS.get(node['class_type'])
            if cls is None or node['class_type'].endswith(('Loader','Encoder')):continue
            fields=cls.INPUT_TYPES()
            required=set(fields.get('required',{}));allowed=required|set(fields.get('optional',{}))
            assert required<=set(node['inputs'])<=allowed, (path,node)
    for path in (ROOT/'workflows').glob('*.ui.json'):
        graph=json.loads(path.read_text())
        assert migrate_ui(graph)==(graph,[])


@pytest.mark.parametrize("keep",[False,True])
def test_schema5_tuning_retention_moves_to_loader_without_shifting_controls(keep):
    graph={"1":dict(class_type="PowerShardConfigTuning",inputs=dict(config=["0",0],keep_workers=keep,
                    reserve_gib=2.,prefetch_blocks=0,numa_policy="auto",strict_attention=True,allow_host_wrappers=False,pin_memory=False)),
           "2":dict(class_type="PowerShardH3Loader",inputs=dict(checkpoint="h3",config=["1",0]))}
    result,changes=migrate_api(graph)
    assert result["2"]["inputs"]["keep_in_memory"]==keep
    assert "keep_workers" not in result["1"]["inputs"] and changes
    ui=dict(nodes=[dict(id=1,type="PowerShardConfigTuning",widgets_values=[2.,0,"auto",keep,True,False,False]),
                   dict(id=2,type="PowerShardH3Loader",widgets_values=["h3"])],links=[[1,1,0,2,0,"POWERSHARD_CONFIG"]])
    result,changes=migrate_ui(ui)
    assert result["nodes"][0]["widgets_values"]==[2.,0,"auto",True,False,False,"fp32","auto"]
    assert result["nodes"][1]["widgets_values"]==["h3",keep] and changes
    assert migrate_ui(result)==(result,[])
