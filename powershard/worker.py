"""Не импортировать из nodes: точка входа отдельного Python process."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import threading
import signal
from datetime import timedelta


def main():
    settings = json.loads(Path(sys.argv[1]).read_text())
    rank = int(sys.argv[2])
    # Watchdog также работает при SIGKILL host. Никаких чужих PIDs.
    parent_pid = settings["parent_pid"]
    def watch_parent():
        while os.getppid() == parent_pid:
            time.sleep(.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch_parent, daemon=True).start()
    protocol = sys.stdout
    sys.stdout = sys.stderr  # Сообщения библиотек не должны попадать в JSON protocol.
    def respond(value):
        protocol.write(json.dumps(value) + "\n"); protocol.flush()
    try:
        from .config import DistributedConfig
        from .patch_config import H3PatchConfig
        from .topology import configure_worker_affinity, process_memory
        config = DistributedConfig(**settings["config"])
        world = len(settings["gpu_uuids"])
        patch = H3PatchConfig(**settings.get("patch", {}))
        locality = configure_worker_affinity(settings["gpu_uuids"][rank], config.numa_policy)
        import torch
        import torch.distributed as dist
        from .preflight import collectives
        from .fsdp_backend import H3Backend, memory
        from .wire import read_payload, write_payload
        torch.set_num_threads(max(1, min(8, len(os.sched_getaffinity(0)))))
        if torch.cuda.device_count() != world:
            raise RuntimeError("CUDA-visible mapping worker не совпал с выбранным набором")
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        cap = torch.cuda.get_device_capability(rank)
        if cap == (7, 0) and "sm_70" not in torch.cuda.get_arch_list():
            print("WARNING: sm_70 не объявлен сборкой; проверяем реальное вычисление", file=sys.stderr)
        if not dist.is_nccl_available():
            raise RuntimeError("NCCL backend отсутствует в PyTorch")
        dist.init_process_group("nccl", rank=rank, world_size=world,
                                init_method=Path(settings["session_dir"], "nccl-store").as_uri(),
                                timeout=timedelta(seconds=config.timeout_s), device_id=device)
        check = collectives(device)
        # FSDP/INT8/repeated no-backward smoke ДО тяжёлых H3 weights.
        from .validation import distributed_probe
        check["fsdp"] = distributed_probe(device, cpu_offload=config.cpu_offload,
                                           pin_memory=config.pin_memory, prefetch_blocks=config.prefetch_blocks, expected_world=world)
        check["locality"] = locality
        backend = None
        if not settings["probe_only"]:
            sys.path.insert(0, settings["comfy_path"])
            sys.argv = ["powershard-worker", "--disable-dynamic-vram", "--use-pytorch-cross-attention"]
            import comfy.options
            comfy.options.enable_args_parsing()
            from .source_guard import verify_comfy
            check["capabilities"] = verify_comfy(settings["comfy_path"])
            start = time.perf_counter()
            if settings.get("role","h3")=="qwen":
                from .qwen_backend import QwenBackend
                backend=QwenBackend(settings["checkpoint"],config,device,patch,settings.get("attention_policy"),settings.get("role_options"))
            else:
                backend = H3Backend(settings["checkpoint"], config, device, patch, settings.get("attention_policy"),settings.get("role_options"))
            check["role"]=settings.get("role","h3")
            check["attention"] = backend.attention.report()
            check["load_s"] = time.perf_counter()-start
            check["checkpoint"] = backend.identity
            check["config"] = config.to_dict()
            check["numeric_policy"] = config.numeric_policy(patch)
            if check["role"]=="qwen":
                check["numeric_policy"].update(compute_dtype="FP16 scaled Linear GEMM; FP32 norm/SiLU/residual/vision patch+attention",
                    residual_dtype="float32",activation_dtype="FP32 restored; bounded FP16 GEMM operands",
                    communication_dtype="FSDP parameter dtype; sequence KV/hidden FP32")
            check["torch"] = torch.__version__
            check["torch_cuda"] = torch.version.cuda
            check["compute_capability"] = cap
            check["shards"] = backend.shards
            check["loaded_memory"] = backend.loaded_memory
            check["cpu_memory"] = process_memory()
            check["patch_fingerprint"] = backend.root.network._ps_patch_fingerprint
            check["patch"] = patch.to_dict()
            check["patch_applied_modules"] = sum(getattr(m, "_ps_safe", False) for m in backend.root.modules())
            qwen=check["role"]=="qwen"
            print(json.dumps({"rank": rank, "GPU": torch.cuda.get_device_name(rank), "capability": cap,"role":check["role"],
                "H3_FP16_Safe": patch.active if not qwen else "not_applicable",
                "condition_proj": ("FP32" if patch.active else "legacy") if not qwen else "native_Qwen_CLIP",
                "residual": "FP32" if patch.active or qwen else "FP16", "MLP_GEMM": "FP16 scaled" if patch.active or qwen else "legacy",
                "spectrum":backend.spectrum.report() if hasattr(backend,"spectrum") else None,
                "attention_requested": config.requested_attention, "attention_effective": backend.attention.effective,
                "attention_dispatch": check["attention"]["dispatch"], "attention_provider": check["attention"]["provider"],
                "attention_fallback_reason": check["attention"]["policy"].get("reason"),
                "world_size": world, "rank_mapping": settings.get("selected_devices", []),
                "parameter_quantization": config.precision, "cpu_offload": config.cpu_offload,
                "prefetch_blocks": config.prefetch_blocks, "patch_fingerprint": check["patch_fingerprint"],
                "memory_settings": config.memory_settings(),
                "persistent_shard_bytes": backend.shard_bytes, "GPU_memory": backend.loaded_memory}, ensure_ascii=False), file=sys.stderr)
        from .diagnostics import loaded_nccl
        check["loaded_nccl"] = loaded_nccl()
        if settings["config"].get("weight_placement") == "ats":
            check["ats"] = settings["ats_diagnostics"][rank]
            print("ATS diagnostic (isolated, before workers): " + json.dumps(check["ats"],ensure_ascii=False),file=sys.stderr)
        check["uuid"] = str(getattr(torch.cuda.get_device_properties(rank), "uuid", "unknown"))
        check["rank_mapping"] = settings.get("selected_devices", [])
        check["world_size"] = world
        respond({"sequence": 0, "preflight": check})
        expected = 1
        # Run-stage device cache: staged conditioning-тензоры читаются с CPU
        # и переносятся на device ОДИН раз за run; последующие шаги берут
        # готовые GPU-тензоры. Кэш сбрасывается, когда меняется stage-каталог.
        stage_cache = {"dir": None, "tensors": {}}
        captured_forward = False
        for line in sys.stdin:
            req = json.loads(line)
            if req["command"] == "shutdown":
                break
            if req.get("sequence") != expected:
                raise RuntimeError("Нарушен порядковый номер команды")
            expected += 1
            if backend is None:
                raise RuntimeError("Сессия запущена только для диагностики")
            if req["command"]=="idle":
                stage_cache["tensors"].clear();stage_cache["dir"]=None
                respond({"sequence":req["sequence"],"phase_offload":backend.idle()})
                continue
            if req["command"]=="end_run":
                captured_forward = False
                report=backend.spectrum.report() if hasattr(backend,"spectrum") else None
                if hasattr(backend,"spectrum"):backend.spectrum.clear()
                stage_cache["tensors"].clear();stage_cache["dir"]=None
                reply={"sequence":req["sequence"],"spectrum_end_run":report}
                if config.min_vram:
                    # Раз в конце sampling, не в каждом diffusion block/step.
                    reply["phase_offload"]=backend.idle()
                respond(reply)
                continue
            stage_dir = req.get("stage") or None
            if stage_dir != stage_cache["dir"]:
                stage_cache["dir"] = stage_dir
                stage_cache["tensors"] = {}
            start = time.perf_counter()
            before_input=memory(device)
            torch.cuda.reset_peak_memory_stats(device)
            payload = read_payload(req["input"], device, base_directory=stage_dir,
                                   stage_cache=stage_cache["tensors"] if stage_dir else None,
                                   cache_device="cpu" if config.min_vram else device,
                                   cache_limit_bytes=config.stage_cache_mib*2**20 if config.min_vram else None)
            torch.cuda.synchronize(device)
            transfer_s = time.perf_counter()-start
            after_input=memory(device)
            capture=os.environ.get("POWERSHARD_AUDIT_CAPTURE")
            if capture and rank==0 and req["command"]=="forward" and not captured_forward:
                # Explicit audit opt-in: may include private conditioning.
                # No copy/sync/disk write in production unless requested.
                destination=Path(capture)/(Path(settings["session_dir"]).name+f"-forward-{req['sequence']}")
                write_payload(destination,payload)
                captured_forward = True
            if os.environ.get("POWERSHARD_PROFILE") == "1" and req["sequence"] <= 2:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                             record_shapes=True,profile_memory=True) as prof:
                    result, metrics = backend.call(req["command"], payload["args"], payload["kwargs"])
                session_name, sequence = Path(settings["session_dir"]).name, req["sequence"]
                trace = Path(settings["report_dir"]) / f"{session_name}-rank{rank}-seq{sequence}.trace.json"
                prof.export_chrome_trace(str(trace))
                metrics["profiler_trace"] = str(trace)
                metrics["collective_events"] = [dict(name=e.key,count=e.count,cpu_total_us=e.cpu_time_total,
                    device_total_us=getattr(e,"device_time_total",None)) for e in prof.key_averages()
                    if any(word in e.key.lower() for word in ("nccl","all_gather","allgather","all_reduce"))]
                metrics["collective_timing_note"] = "Вложенные/overlapped events; не суммировать с forward wall time"
            else:
                result, metrics = backend.call(req["command"], payload["args"], payload["kwargs"])
            metrics["input_transfer_s"] = transfer_s
            if rank == 0:
                write_payload(req["output"], result)
            rpc_peak=max(after_input["peak_allocated"],torch.cuda.max_memory_allocated(device))
            del result, payload
            metrics["rpc_memory"] = dict(before_input=before_input,after_input=after_input,
                after_result_release=memory(device),peak_allocated_including_input=rpc_peak,
                scope="one RPC; overlapping snapshots, do not add")
            storages={(str(t.device),t.untyped_storage().data_ptr()):t.untyped_storage().nbytes()
                      for t in stage_cache["tensors"].values() if isinstance(t,torch.Tensor)}
            metrics["stage_cache"] = dict(storage_bytes=sum(storages.values()),
                tensors=sum(isinstance(t,torch.Tensor) for t in stage_cache["tensors"].values()),
                residency="cpu" if config.min_vram else str(device),
                limit_bytes=config.stage_cache_mib*2**20 if config.min_vram else None)
            metrics["memory_settings"] = config.memory_settings()
            respond({"sequence": req["sequence"], "metrics": metrics})
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        respond({"error": error})
        return 1
    finally:
        if "dist" in locals() and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
