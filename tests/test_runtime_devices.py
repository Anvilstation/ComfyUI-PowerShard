"""Только Python lifecycle contracts; mock Popen НЕ доказывает NCCL/FSDP."""
import json
from pathlib import Path
import queue
from types import SimpleNamespace
import pytest
from powershard.runtime import Session
from powershard.config import DistributedConfig
from powershard.patch_config import H3PatchConfig
from powershard.attention_policy import prepare_policy


@pytest.mark.parametrize('n',[1,2,3,4,6,9])
def test_spawn_count_order_cleanup(n,tmp_path,monkeypatch):
    inventory=[dict(user_id=str(i),uuid=f'GPU-{i:08x}',name='mock',total_memory=1) for i in range(n)]
    monkeypatch.setattr('powershard.devices.visible_inventory',lambda:inventory)
    monkeypatch.setattr('powershard.attention_policy.prepare_policy',lambda *a,**kw:dict(fingerprint='mock-policy'))
    monkeypatch.setattr('powershard.runtime.threading.Thread',lambda **k:SimpleNamespace(start=lambda:None))
    spawned=[]
    class Process:
        def __init__(self,cmd,**kw):
            self.cmd=cmd;self.env=kw['env'];self.code=None
            self.stdin=SimpleNamespace(write=lambda x:None,flush=lambda:None,close=lambda:None)
            self.stdout=SimpleNamespace(close=lambda:None)
            self.settings=json.loads(Path(cmd[-2]).read_text());spawned.append(self)
        def poll(self):return self.code
        def wait(self,**kw):self.code=0;return 0
        def terminate(self):self.code=-15
        def kill(self):self.code=-9
    monkeypatch.setattr('powershard.runtime.subprocess.Popen',Process)
    monkeypatch.setattr(Session,'_wait',lambda s,*a:[dict(sequence=0) for _ in s.processes])
    config=DistributedConfig(gpu_ids=tuple(str(i) for i in reversed(range(n))))
    session=Session(None,config,tmp_path,tmp_path/'reports',probe_only=True)
    session.start();session.start()
    assert len(spawned)==n and session.running
    expected=[inventory[i]['uuid'] for i in reversed(range(n))]
    for rank,p in enumerate(spawned):
        assert p.cmd[-1]==str(rank) and p.env['CUDA_VISIBLE_DEVICES']==','.join(expected)
        assert p.settings['selected_devices'][rank]['rank']==rank
    other=session.with_patch(H3PatchConfig(enabled=True))
    assert other is not session and not session.patch.active and other.patch.active
    session.close()
    assert all(p.code==0 for p in spawned) and not session.running
    session.start();assert len(spawned)==2*n
    # Программная замена immutable config тоже пересоздаёт настоящую session.
    session.config=DistributedConfig(gpu_ids=('0',),attention_backend='sdpa')
    session.start();assert len(session.processes)==1 and len(spawned)==2*n+1
    session.close();other.close()


def test_rank_error_and_timeout_are_not_suppressed(tmp_path):
    s=Session(None,DistributedConfig(timeout_s=.01),tmp_path,tmp_path)
    s.processes=[SimpleNamespace()]*2;s.responses=[queue.Queue(),queue.Queue()]
    s.responses[1].put(dict(error='device-side assert'))
    with pytest.raises(RuntimeError,match='rank 1'):s._wait(1)
    with pytest.raises(TimeoutError):s._wait(2)
    s.processes=[];s.responses=[]


def test_policy_common_to_all_ranks_and_sage_opt_in(monkeypatch):
    seen=[]
    def probe(provider,device,*args):
        seen.append((provider,device['uuid']))
        return dict(status='FAIL',reason='kernel rejected launch') if provider=='vllm_flash_attn' and device['uuid']=='GPU-b' else dict(status='PASS',identity={'module':provider})
    monkeypatch.setattr('powershard.attention_policy.isolated_probe',probe)
    devices=[dict(uuid='GPU-a'),dict(uuid='GPU-b')]
    with pytest.warns(UserWarning):policy=prepare_policy(DistributedConfig(attention_backend='vllm_flash_attn'),devices)
    assert policy['effective']=='sdpa' and policy['devices']==['GPU-a','GPU-b']
    assert ('vllm_flash_attn','GPU-a') in seen and ('vllm_flash_attn','GPU-b') in seen
    with pytest.raises(RuntimeError):prepare_policy(DistributedConfig(attention_backend='vllm_flash_attn',allow_fallback=False),devices)
    seen.clear();prepare_policy(DistributedConfig(attention_backend='auto'),devices)
    assert all(provider!='sageattention' for provider,_ in seen)
