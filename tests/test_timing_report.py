import json
from powershard.timing_report import extract_forward_rows, summarize


def history():
    return [dict(sequence=0,ranks=[dict(preflight=dict(config=dict(weight_placement="cpu",sequence_comm_dtype="fp32")))]),
            dict(sequence=2,command="forward",ranks=[dict(metrics=dict(forward_s=t,
                memory_plan=dict(estimated_global_tokens=46535,requested_prefetch=2,effective_prefetch=0),
                wall_phases=dict(finish_cuda_wait_s=t/10),sequence_communication=dict(scale_all_reduces=0)))
                for t in (79.,80.,82.,81.,79.5)],wait_all_ranks_s=83.,output_deserialization_s=.01)]


def test_critical_rank_wall_time_is_max_not_sum_and_missing_stays_unknown():
    row=extract_forward_rows(history())[0]
    assert row["rank_count"]==5 and row["forward_s"]==82.
    assert row["finish_cuda_wait_s"]==8.2 and row["wait_all_ranks_s"]==83.
    assert row["output_serialization_s"] is None and row["scale_all_reduces"]==0
    assert row["config"]["sequence_comm_dtype"]=="fp32"


def test_inputs_settings_and_profile_are_not_mixed(tmp_path):
    base=history()
    for i in range(3):
        data=json.loads(json.dumps(base))
        if i==1:data[0]["ranks"][0]["preflight"]["config"]["sequence_comm_dtype"]="fp16"
        if i==2:data[1]["ranks"][0]["metrics"]["profiler_trace"]="trace.json"
        (tmp_path/f"run-{i}.json").write_text(json.dumps(data))
    (tmp_path/"unrelated.json").write_text("{}")
    report=summarize([tmp_path,tmp_path])
    assert len(report["groups"])==len(report["forwards"])==3
    assert all(g["median_s"]["forward_s"]==82. for g in report["groups"])
    assert summarize([])["status"]=="NO_FORWARD_MEASUREMENTS"


def test_session_and_run_wrappers_do_not_double_count_same_rpc(tmp_path):
    data=history()
    data[1]["session_id"]="powershard-test-session"
    (tmp_path/"powershard-test-session.json").write_text(json.dumps(data))
    (tmp_path/"run-example.json").write_text(json.dumps(dict(config=data[0]["ranks"][0]["preflight"]["config"],history=data[1:])))
    report=summarize([tmp_path])
    assert len(report["forwards"])==1 and report["groups"][0]["forward_rpc_count"]==1
