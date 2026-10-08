"""Read-only wall-time extraction. Missing measurements stay UNKNOWN/None."""
import json
import statistics
from pathlib import Path


def history_rows(document):
    if isinstance(document, list):
        return document
    if isinstance(document, dict):
        return document.get("rank_history", document.get("session_history", document.get("history", [])))
    return []


def number_max(values):
    measured = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return max(measured) if measured else None


def extract_forward_rows(document, source=""):
    config = document.get("config", {}) if isinstance(document, dict) else {}
    checkpoint = document.get("checkpoint") if isinstance(document, dict) else None
    rows = []
    for event in history_rows(document):
        if not isinstance(event, dict):
            continue
        ranks = event.get("ranks", [])
        for rank in ranks:
            preflight = rank.get("preflight", {})
            if preflight:
                config = preflight.get("config", config)
                checkpoint = preflight.get("checkpoint", checkpoint)
        metrics = [r.get("metrics", {}) for r in ranks]
        metrics = [m for m in metrics if "forward_s" in m]
        if not metrics or event.get("command") != "forward":
            continue
        first = metrics[0]
        plan = first.get("memory_plan", {})
        communication = first.get("sequence_communication", {})
        row = dict(source=source, session_id=event.get("session_id"), sequence=event.get("sequence"), rank_count=len(metrics),
            config=config, checkpoint=checkpoint,
            effective_attention=first.get("attention", {}).get("effective"),
            tokens=plan.get("estimated_global_tokens"),
            requested_prefetch=plan.get("requested_prefetch"), effective_prefetch=plan.get("effective_prefetch"),
            prefetch_reason=plan.get("prefetch_reason"),
            forward_s=number_max(m.get("forward_s") for m in metrics),
            worker_request_s=number_max(m.get("worker_request_s") for m in metrics),
            input_transfer_s=number_max(m.get("input_transfer_s") for m in metrics),
            output_serialization_s=number_max(m.get("output_serialization_s") for m in metrics),
            wait_all_ranks_s=event.get("wait_all_ranks_s"),
            output_deserialization_s=event.get("output_deserialization_s"),
            input_serialization_s=event.get("input_serialization_s"),
            ipc_and_forward_s=event.get("ipc_and_forward_s"),
            resume_from_ram_s=number_max(m.get("phase_cache", {}).get("resume_from_ram_s") for m in metrics),
            scale_all_reduces=communication.get("scale_all_reduces"),
            effective_wire_dtype=communication.get("communication_dtype"),
            profiler_enabled=any("profiler_trace" in m for m in metrics),
            cuda_peak_allocated_bytes=number_max(m.get("memory", {}).get("peak_allocated") for m in metrics),
            mlp_chunks=sorted({v.get("effective_tokens") for m in metrics for v in m.get("mlp", {}).values()
                               if v.get("effective_tokens") is not None}))
        for field in ("entry_sync_s", "memory_plan_s", "compute_enqueue_and_reshard_s", "finish_cuda_wait_s", "finite_validation_s"):
            row[field] = number_max(m.get("wall_phases", {}).get(field) for m in metrics)
        row["progress_write_wall_s"] = number_max(m.get("progress_io", {}).get("write_wall_s") for m in metrics)
        rows.append(row)
    return rows


def summarize(paths):
    rows, errors = [], []
    seen = set()
    for path in paths:
        path = Path(path)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file in files:
            if file.resolve() in seen:
                continue
            seen.add(file.resolve())
            try:
                rows.extend(extract_forward_rows(json.loads(file.read_text()), str(file)))
            except (OSError, ValueError, TypeError, KeyError) as error:
                errors.append(dict(path=str(file), error=str(error)))
    groups = {}
    unique=[];seen_rpc=set()
    for row in rows:
        if row["session_id"] is not None:
            key=(row["session_id"],row["sequence"])
            if key in seen_rpc:continue
            seen_rpc.add(key)
        unique.append(row)
        # Never mix shapes/settings/profiler runs. Number of RPCs is not the
        # number of sampler iterations (CFG/conditioning can make them differ).
        signature = json.dumps(dict(config=row["config"], checkpoint=row["checkpoint"],
            tokens=row["tokens"], rank_count=row["rank_count"], attention=row["effective_attention"],
            effective_prefetch=row["effective_prefetch"], profiler_enabled=row["profiler_enabled"]), sort_keys=True)
        groups.setdefault(signature, []).append(row)
    summaries = []
    for signature, items in groups.items():
        record = dict(signature=json.loads(signature), forward_rpc_count=len(items))
        record["median_s"] = {}
        for key in items[0]:
            if key.endswith("_s"):
                values = [r[key] for r in items if isinstance(r.get(key), (int, float))]
                record["median_s"][key] = statistics.median(values) if values else None
        summaries.append(record)
    return dict(status="MEASURED_WALL_TIMES" if unique else "NO_FORWARD_MEASUREMENTS",
                groups=summaries, forwards=unique, errors=errors,
                note="Max across ranks per RPC; medians include all collected calls, not a cold/warm classification. None=not measured. Wall phases overlap GPU work; do not sum them into GPU execution time. One RPC is not necessarily one sampler iteration.")
