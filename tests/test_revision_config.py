import pytest
from powershard.config import DistributedConfig
from powershard.nodes import PowerShardConfig, PowerShardConfigTuning, PowerShardH3MLP, PowerShardQwenMLP
from powershard.ats_memory import ATTRIBUTES


def test_three_distinct_memory_modes():
    for placement in ("gpu", "cpu", "ats"):
        config = DistributedConfig(weight_placement=placement)
        assert config.cpu_offload == (placement == "cpu")
        assert DistributedConfig(**config.to_dict()) == config
    assert DistributedConfig(cpu_offload=True).weight_placement == "cpu"
    assert not DistributedConfig(weight_placement="ats",cpu_offload=True).cpu_offload


def test_minimal_config_and_legacy_conversion():
    fields = PowerShardConfig.INPUT_TYPES()
    assert list(fields["required"]) == ["gpu_ids","weight_placement","precision","attention_backend","sequence_mode"]
    assert "optional" not in fields
    config = PowerShardConfig().create("all", "cpu", "fp16", "auto", "ulysses")[0]
    assert config.cpu_offload and config.backend == "fsdp2_sequence"
    assert "timeout_s" not in config.to_dict() and "allow_unverified" not in config.to_dict()
    with pytest.warns(UserWarning):
        assert DistributedConfig(backend="fsdp2",timeout_s=1).backend == "fsdp2_sequence"
    tuned = PowerShardConfigTuning().tune(config,keep_workers=True)[0]
    assert not tuned.release_after_sampling
    assert "mlp_chunk_tokens" not in PowerShardConfig.INPUT_TYPES()["required"]
    assert PowerShardH3MLP.INPUT_TYPES()["required"]["model"] == ("MODEL",)
    assert PowerShardQwenMLP.INPUT_TYPES()["required"]["clip"] == ("CLIP",)


def test_ats_requires_host_page_tables_not_just_uva(monkeypatch):
    import ctypes
    from powershard import ats_memory
    assert ATTRIBUTES["unified_addressing"] == 41  # CUDA DRIVER enum
    class Driver:
        def cuDeviceGetAttribute(self, pointer, attr, index):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 0 if attr == 100 else 1
            return 0
    monkeypatch.setattr(ats_memory, "driver_api", lambda: Driver())
    result = ats_memory.ats_capabilities(0)
    assert result["attributes"]["unified_addressing"] == 1
    assert not result["ats_supported"] and result["status"] == "FAIL"


def test_custom_attention_probe_includes_padded_local_heads(monkeypatch):
    import powershard.attention_policy as policy
    import powershard.checkpoint as checkpoint
    class TinyCheckpoint:
        def __init__(self,path):pass
        def model_config(self):return dict(attention_head_dim=128,num_attention_heads=56)
    monkeypatch.setattr(checkpoint,"Checkpoint",TinyCheckpoint)
    seen=[]
    def probe(provider,device,dim,heads,timeout,kv_heads):
        seen.append((device['uuid'],heads))
        return dict(status='PASS',identity={'module':'test'},uuid=device['uuid'])
    monkeypatch.setattr(policy,'isolated_probe',probe)
    devices=[{'uuid':f'GPU-{i}'} for i in range(3)]
    result=policy.prepare_policy(DistributedConfig(attention_backend='flash_attn',sequence_mode='ulysses'),devices,'tiny')
    assert result['geometry']['certified_heads']==[56,19]
    assert seen==[(f'GPU-{i}',h) for i in range(3) for h in (56,19)]


def test_auto_attention_tries_custom_provider_before_sdpa(monkeypatch):
    import powershard.attention_policy as policy
    seen=[]
    def probe(provider,*args):
        seen.append(provider)
        return dict(status='PASS',identity={'module':'test'})
    monkeypatch.setattr(policy,'isolated_probe',probe)
    result=policy.prepare_policy(DistributedConfig(attention_backend='auto'),[dict(uuid='GPU-0')])
    assert seen==['vllm_flash_attn'] and result['effective']=='vllm_flash_attn'


def test_missing_flash_extension_checks_installed_vllm_fallback(monkeypatch):
    import powershard.attention_policy as policy
    seen=[]
    def probe(provider,*args):
        seen.append(provider)
        return dict(status='FAIL',reason="No module named flash_attn_2_cuda") if provider=='flash_attn' else dict(status='PASS',identity={'module':'test'})
    monkeypatch.setattr(policy,'isolated_probe',probe)
    with pytest.warns(UserWarning,match='Effective: vllm_flash_attn'):
        result=policy.prepare_policy(DistributedConfig(attention_backend='flash_attn'),[dict(uuid='GPU-0')])
    assert seen==['flash_attn','vllm_flash_attn'] and result['effective']=='vllm_flash_attn'
    seen.clear()
    with pytest.raises(RuntimeError):
        policy.prepare_policy(DistributedConfig(attention_backend='flash_attn',allow_fallback=False),[dict(uuid='GPU-0')])
    assert seen==['flash_attn']


def test_ats_native_pool_is_worker_local_and_preserves_allocator_options():
    from powershard.ats_memory import ats_worker_environment
    original=dict(CUDA_VISIBLE_DEVICES='GPU-abc',PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync,expandable_segments:True,roundup_power2_divisions:[256:1,512:2]')
    result=ats_worker_environment(original)
    assert result['CUDA_VISIBLE_DEVICES']==original['CUDA_VISIBLE_DEVICES']
    assert 'cudaMallocAsync' in original['PYTORCH_CUDA_ALLOC_CONF']
    assert result['PYTORCH_ALLOC_CONF']==result['PYTORCH_CUDA_ALLOC_CONF']=='roundup_power2_divisions:[256:1,512:2],backend:native,expandable_segments:False'
