import json
import io
from types import SimpleNamespace
import pytest
from powershard.telemetry import RequestProgress,profiler_summary


def test_cancelled_rpc_preserves_chosen_mlp_and_failure_status(tmp_path,monkeypatch):
    from powershard.runtime import Session
    from powershard.config import DistributedConfig
    session=Session(None,DistributedConfig(),tmp_path,tmp_path/"reports")
    session.path=tmp_path/"powershard-test";session.path.mkdir()
    progress=RequestProgress(session.report_dir/"powershard-test-rank0-progress.json",rank=0)
    progress.begin(1,"forward")
    progress("mlp_plan",dict(effective_tokens=256,local_tokens=9307))
    session.processes=[SimpleNamespace(stdin=io.StringIO())]
    monkeypatch.setattr(session,"start",lambda cancel:None)
    monkeypatch.setattr(session,"close",lambda **kwargs:None)
    def interrupt(sequence,cancel):raise KeyboardInterrupt()
    monkeypatch.setattr(session,"_wait",interrupt)
    with pytest.raises(KeyboardInterrupt):session.call("forward",(),{})
    event=session.history[-1]
    assert event["status"]=="INTERRUPTED_OR_FAILED"
    assert event["rank_progress"][0]["mlp_plan"]["effective_tokens"]==256
    assert not list(session.report_dir.glob("*.tmp"))


def test_profile_distinguishes_operator_selection_from_kernel_observation():
    class Profile:
        def events(self):return [SimpleNamespace(device_type="DeviceType.CUDA",name="volta_hgemm",device_time_total=13),
                                SimpleNamespace(device_type="DeviceType.CUDA",name="flash_fwd",device_time_total=8),
                                SimpleNamespace(device_type="DeviceType.CPU",name="aten::linear",device_time_total=0)]
        def key_averages(self):return [SimpleNamespace(key="aten::_scaled_dot_product_efficient_attention",count=50)]
    result=profiler_summary(Profile())
    assert result["attention_operators"]=={"aten::_scaled_dot_product_efficient_attention":50}
    assert result["attention_kernel_names"]==["flash_fwd"]
    assert result["cuda_kernel_count"]==2 and result["triton_kernel_names"]==[]
