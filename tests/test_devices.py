import pytest
from powershard.config import DistributedConfig
from powershard.devices import resolve_gpu_selection
from powershard.nodes import PowerShardConfig


def inventory(n=12):
    # CUDA_VISIBLE_DEVICES=GPU-9,GPU-3,... уже отражён в CUDA visible inventory.
    return [dict(user_id=str(i),uuid=f'GPU-{i+100:08x}',name=f'GPU {i}',total_memory=(i+1)*2**30) for i in range(n)]


@pytest.mark.parametrize('n',[1,2,3,4,5,6,12])
def test_any_world(n):
    selected=resolve_gpu_selection(','.join(map(str,range(n))),inventory())
    assert len(selected)==n and [x['rank'] for x in selected]==list(range(n))
    assert [x['worker_cuda_index'] for x in selected]==list(range(n))


def test_order_all_visible_and_duplicates(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','5,2,0,4,1,3')
    inv=inventory(6)
    selected=resolve_gpu_selection('5,2,0',inv)
    assert [x['uuid'] for x in selected]==[inv[i]['uuid'] for i in (5,2,0)]
    assert len(resolve_gpu_selection('all',inv))==6
    with pytest.warns(UserWarning):assert len(resolve_gpu_selection('1,1,2',inv))==2
    with pytest.warns(UserWarning):assert len(resolve_gpu_selection(('1',inv[1]['uuid']),inv))==1
    for value in ('','-1','9','0,','all,1'):
        with pytest.raises(ValueError):resolve_gpu_selection(value,inv)
    with pytest.raises(ValueError):resolve_gpu_selection('all',[])


def test_legacy_widgets_order_and_new_optional_defaults():
    fields=PowerShardConfig.INPUT_TYPES()
    assert list(fields['required'])==['gpu_ids','weight_placement','precision','attention_backend','sequence_mode']
    old=PowerShardConfig().create(gpu_ids='0,1,2',precision='fp16',attention_backend='math',backend='fsdp2',timeout_s=600)[0]
    assert old.requested_attention=='math' and old.allow_fallback
    for count in (3,2,6):
        c=DistributedConfig(gpu_ids=tuple(map(str,range(count))),cpu_offload=True,attention_backend='sdpa')
        assert DistributedConfig(**c.to_dict())==c
        assert len(resolve_gpu_selection(c.gpu_ids,inventory()))==count


def test_inventory_subprocess_from_comfy_cwd(tmp_path,monkeypatch):
    # Реальный subprocess, скрытые CUDA devices: без GPU kernel/workers.
    from powershard.devices import visible_inventory
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    assert visible_inventory()==[]
