import pytest
from powershard.config import DistributedConfig
from powershard.topology import cpu_list,numa_launch_prefix,process_memory


def test_offload_prefetch_config():
    c=DistributedConfig(cpu_offload=True,pin_memory=False,prefetch_blocks=2,numa_policy='auto')
    assert DistributedConfig(**c.to_dict())==c
    assert DistributedConfig().cpu_offload is False
    for count in (-1,3):
        with pytest.raises(ValueError):DistributedConfig(prefetch_blocks=count)
    with pytest.raises(ValueError):DistributedConfig(numa_policy='guess')


def test_linux_topology_helpers():
    assert cpu_list('0-3,8,10-12')==[0,1,2,3,8,10,11,12]
    assert numa_launch_prefix('unused','none')==[]
    assert process_memory()['VmRSS_bytes']>0


def test_all_workflows_patch_both_model_consumers():
    import json
    from pathlib import Path
    folder=Path(__file__).resolve().parents[1]/'workflows'
    for filename in folder.glob('*.api.json'):
        graph=json.loads(filename.read_text())
        assert graph['19']['class_type']=='PowerShardH3FP16Patcher'
        assert graph['19']['inputs']['enabled']
        model=['19',0]
        if '22' in graph:
            assert graph['22']['class_type']=='PowerShardSpectrum'
            assert graph['22']['inputs']['model']==model
            model=['22',0]
        for node in ('7','10'):assert graph[node]['inputs']['model']==model
        if graph['3']['class_type']=='PowerShardH3TextEncoder':
            assert graph['1']['inputs']['cpu_offload']==('offload' in filename.name)
        elif graph['3']['inputs']['idle_policy']=='cpu_shards':
            config_id=graph['3']['inputs']['config'][0]
            assert graph[config_id]['inputs']['cpu_offload']
