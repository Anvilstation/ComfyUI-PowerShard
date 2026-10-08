"""Точка входа отдельного Wan rank-процесса. Протокол совпадает с powershard.worker.

Отличия: роль ``wan``, несколько экспертов (MoE high/low) в одном process group,
capability-проверка comfy.ldm.wan вместо MiniMax H3. Существующий worker.py не меняется.
"""
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from datetime import timedelta


def verify_wan_comfy(comfy_path):
    """Capabilities, а не версия ComfyUI."""
    import importlib
    from .source_guard import require_signature
    report = {"admission": "capabilities_only", "signatures": {}}
    model = importlib.import_module("comfy.ldm.wan.model")
    for label, function, arguments in (
            ("WanModel.__init__", model.WanModel.__init__, ("dim", "num_heads", "operations", "dtype", "device")),
            ("WanModel.rope_encode", model.WanModel.rope_encode, ("t", "h", "w", "device", "dtype", "transformer_options")),
            ("WanModel.unpatchify", model.WanModel.unpatchify, ("x", "grid_sizes")),
            ("sinusoidal_embedding_1d", model.sinusoidal_embedding_1d, ("dim", "position"))):
        report["signatures"][label] = require_signature(function, arguments, label)
    common = importlib.import_module("comfy.ldm.common_dit")
    report["signatures"]["pad_to_patch_size"] = require_signature(common.pad_to_patch_size, ("img", "patch_size"),
                                                                  "pad_to_patch_size")
    report["comfy_path"] = str(comfy_path)
    return report


def verify_ltx_comfy():
    """Capabilities родного LTX (comfy.ldm.lightricks), которые использует PowerShard."""
    import importlib
    from .source_guard import require_signature
    model = importlib.import_module("comfy.ldm.lightricks.model")
    av = importlib.import_module("comfy.ldm.lightricks.av_model")
    report = {}
    for label, function, arguments in (
            ("LTXBaseModel._precompute_freqs_cis", model.LTXBaseModel._precompute_freqs_cis,
             ("indices_grid", "dim", "out_dtype", "max_pos", "use_middle_indices_grid", "num_attention_heads")),
            ("LTXVModel._build_guide_self_attention_mask", model.LTXVModel._build_guide_self_attention_mask,
             ("x", "transformer_options", "merged_args")),
            ("LTXVModel.keyframes_abs_pos_mask", model.LTXVModel.keyframes_abs_pos_mask,
             ("pixel_coords", "orig_shape", "grid_mask", "num_guide_tokens", "generated_keyframes")),
            ("compute_prompt_timestep", model.compute_prompt_timestep,
             ("adaln_module", "timestep_scaled", "batch_size", "hidden_dtype")),
            ("LTXAVModel._prepare_context", av.LTXAVModel._prepare_context, ("context", "batch_size", "x", "attention_mask")),
            ("LTXAVModel.preprocess_text_embeds", av.LTXAVModel.preprocess_text_embeds, ("context", "unprocessed"))):
        report[label] = require_signature(function, arguments, label)
    for name in ("GuideAttentionMask", "CrossAttention", "FeedForward", "AdaLayerNormSingle"):
        if not hasattr(model, name):
            raise RuntimeError(f"LTX capability: comfy.ldm.lightricks.model.{name} отсутствует")
    if not hasattr(av, "BasicAVTransformerBlock"):
        raise RuntimeError("LTX capability: BasicAVTransformerBlock отсутствует")
    return report


def main():
    settings = json.loads(Path(sys.argv[1]).read_text())
    rank = int(sys.argv[2])
    parent_pid = settings["parent_pid"]

    def watch_parent():
        while os.getppid() == parent_pid:
            time.sleep(.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch_parent, daemon=True).start()
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def respond(value):
        protocol.write(json.dumps(value) + "\n")
        protocol.flush()
    try:
        from .config import DistributedConfig
        from .patch_config import H3PatchConfig
        from .wan_config import WanOptions
        from .topology import configure_worker_affinity, process_memory
        config = DistributedConfig(**settings["config"])
        world = len(settings["gpu_uuids"])
        patch = H3PatchConfig(**settings.get("patch", {}))
        role_options = settings.get("role_options", {})
        kind = role_options.get("kind")
        options = WanOptions(**role_options.get("options", {})) if kind not in ("ltx", "native_te", "h3q") else None
        locality = configure_worker_affinity(settings["gpu_uuids"][rank], config.numa_policy)
        import torch
        import torch.distributed as dist
        from .preflight import collectives
        from .wire import read_payload, write_payload
        torch.set_num_threads(max(1, min(8, len(os.sched_getaffinity(0)))))
        if torch.cuda.device_count() != world:
            raise RuntimeError("CUDA-visible mapping worker не совпал с выбранным набором")
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        from .fp16_safe import configure_matmul
        # Scaled Safe GEMM ограничивает суммы; в быстром режиме это тот же default,
        # что у PyTorch/ComfyUI. Full-half accumulation остаётся выключенной.
        configure_matmul(True)
        managed_pool = None
        if config.weight_placement == "ats":
            from .ats_memory import ManagedShardPool
            managed_pool = ManagedShardPool(device)
        cap = torch.cuda.get_device_capability(rank)
        if cap == (7, 0) and "sm_70" not in torch.cuda.get_arch_list():
            print("WARNING: sm_70 не объявлен сборкой; проверяем реальное вычисление", file=sys.stderr)
        if not dist.is_nccl_available():
            raise RuntimeError("NCCL backend отсутствует в PyTorch")
        dist.init_process_group("nccl", rank=rank, world_size=world,
                                init_method=Path(settings["session_dir"], "nccl-store").as_uri(),
                                timeout=timedelta(days=365), device_id=device)
        check = collectives(device)
        from .validation import distributed_probe
        check["fsdp"] = distributed_probe(device, cpu_offload=config.cpu_offload, pin_memory=config.pin_memory,
                                          prefetch_blocks=config.prefetch_blocks, expected_world=world,
                                          managed_pool=managed_pool)
        if managed_pool is not None:
            check["ats"] = managed_pool.report()
        check["locality"] = locality
        backend = None
        if not settings["probe_only"]:
            sys.path.insert(0, settings["comfy_path"])
            sys.argv = ["powershard-wan-worker", "--disable-dynamic-vram", "--use-pytorch-cross-attention"]
            import comfy.options
            comfy.options.enable_args_parsing()
            check["capabilities"] = verify_wan_comfy(settings["comfy_path"])
            start = time.perf_counter()
            from .wan_backend import WanBackend, WanExperts
            from .phase_cache import ResidentBackend
            experts = role_options["experts"]
        if not settings["probe_only"] and role_options.get("kind") == "t5":
            from .wan_text import WanT5Backend

            def make_t5(local_state, initial_pool=None):
                pool = initial_pool
                if config.weight_placement == "ats" and pool is None:
                    from .ats_memory import ManagedShardPool
                    pool = ManagedShardPool(device)
                return WanT5Backend(experts["t5"], config, device, pool, local_state)
            backend = ResidentBackend(make_t5(None, managed_pool), make_t5, device)
            check["role"] = "wan_t5"
            check["experts"] = {"t5": dict(identity=backend._backend.identity, lora=[])}
            check["attention"] = {"policy": settings["attention_policy"], "effective_used": "comfy_native_fp32"}
            check["load_s"] = time.perf_counter() - start
            check["load_timing"] = {"t5": backend._backend.load_timing}
            check["config"] = config.to_dict()
            check["torch"] = torch.__version__
            check["compute_capability"] = cap
            check["loaded_memory"] = {"t5": backend._backend.loaded_memory}
            check["shard_bytes"] = {"t5": backend._backend.shard_bytes}
            check["cpu_memory"] = process_memory()
            check["patch_fingerprint"] = patch.fingerprint()
            check["wan_fingerprints"] = {"t5": backend._backend.patch_fingerprint}
            print(json.dumps({"rank": rank, "role": "wan_t5", "world_size": world, "weight_placement": config.weight_placement,
                              "shard_bytes": check["shard_bytes"], "load_s": round(check["load_s"], 2),
                              "load_timing": check["load_timing"]}, ensure_ascii=False), file=sys.stderr)
        elif not settings["probe_only"] and kind == "h3q":
            from .h3_quant import quant_h3_backend
            from .source_guard import verify_comfy
            check["h3_capabilities"] = verify_comfy(settings["comfy_path"])
            backend_class = quant_h3_backend()

            def make_h3(local_state, initial_pool=None):
                pool = initial_pool
                if config.weight_placement == "ats" and pool is None:
                    from .ats_memory import ManagedShardPool
                    pool = ManagedShardPool(device)
                return backend_class(experts["main"], config, device, patch, settings.get("attention_policy"),
                                     role_options, pool, local_state)
            backend = ResidentBackend(make_h3(None, managed_pool), make_h3, device)
            live = backend._backend
            check["role"] = "h3"
            check["experts"] = {"main": dict(identity=live.identity, lora=[])}
            check["attention"] = live.attention.report()
            check["load_s"] = time.perf_counter() - start
            check["load_timing"] = {"main": live.load_timing}
            check["quantization"] = live.quant_report
            check["config"] = config.to_dict()
            check["numeric_policy"] = config.numeric_policy(patch)
            check["torch"] = torch.__version__
            check["compute_capability"] = cap
            check["loaded_memory"] = {"main": live.loaded_memory}
            check["shard_bytes"] = {"main": live.shard_bytes}
            check["cpu_memory"] = process_memory()
            check["patch_fingerprint"] = patch.fingerprint()
            check["wan_fingerprints"] = {"main": live.patch_fingerprint}
            print(json.dumps({"rank": rank, "GPU": torch.cuda.get_device_name(rank), "role": "h3 (quant loader)",
                              "quantization": live.quant_report, "world_size": world,
                              "weight_placement": config.weight_placement, "shard_bytes": check["shard_bytes"],
                              "load_s": round(check["load_s"], 2)}, ensure_ascii=False, default=str), file=sys.stderr)
        elif not settings["probe_only"] and kind == "native_te":
            from .native_te import NativeTEBackend

            def make_te(local_state, initial_pool=None):
                pool = initial_pool
                if config.weight_placement == "ats" and pool is None:
                    from .ats_memory import ManagedShardPool
                    pool = ManagedShardPool(device)
                return NativeTEBackend(role_options["files"], role_options["clip_type"], config, device, pool, local_state,
                                       role_options.get("options"))
            backend = ResidentBackend(make_te(None, managed_pool), make_te, device)
            check["role"] = "native_te"
            check["quantization"] = backend._backend.quant_report
            check["experts"] = {"te": dict(identity=backend._backend.identity, lora=[])}
            check["attention"] = {"policy": settings["attention_policy"], "effective_used": "comfy_native_fp32"}
            check["load_s"] = time.perf_counter() - start
            check["load_timing"] = {"te": backend._backend.load_timing}
            check["config"] = config.to_dict()
            check["torch"] = torch.__version__
            check["compute_capability"] = cap
            check["loaded_memory"] = {"te": backend._backend.loaded_memory}
            check["shard_bytes"] = {"te": backend._backend.shard_bytes}
            check["cpu_memory"] = process_memory()
            check["patch_fingerprint"] = patch.fingerprint()
            check["wan_fingerprints"] = {"te": backend._backend.patch_fingerprint}
            print(json.dumps({"rank": rank, "role": "native_te", "world_size": world, "encoder": backend._backend.identity,
                              "shard_bytes": check["shard_bytes"], "load_s": round(check["load_s"], 2),
                              "load_timing": check["load_timing"]}, ensure_ascii=False, default=str), file=sys.stderr)
        elif not settings["probe_only"] and kind == "ltx":
            from .ltx_backend import LTXBackend
            from .ltx_config import LTXOptions
            ltx_options = LTXOptions(**role_options.get("options", {}))
            check["ltx_capabilities"] = verify_ltx_comfy()

            def make_ltx(local_state, initial_pool=None):
                pool = initial_pool
                if config.weight_placement == "ats" and pool is None:
                    from .ats_memory import ManagedShardPool
                    pool = ManagedShardPool(device)
                return LTXBackend(experts["main"], config, device, ltx_options, settings.get("attention_policy"),
                                  tuple(role_options.get("loras", {}).get("main", ())), pool, local_state)
            backend = ResidentBackend(make_ltx(None, managed_pool), make_ltx, device)
            live = backend._backend
            check["role"] = "ltx"
            check["quantization"] = live.quant_report
            check["experts"] = {"main": dict(identity=live.identity, lora=live.lora_report)}
            check["attention"] = live.attention.report()
            check["load_s"] = time.perf_counter() - start
            check["load_timing"] = {"main": live.load_timing}
            check["config"] = config.to_dict()
            check["numeric_policy"] = dict(config.numeric_policy(None),
                                           compute_dtype=("FP16 scaled GEMM (Safe)" if ltx_options.fp16_safe else ltx_options.weight_dtype.upper() + " GEMM")
                                           + "; FP32 residual/AdaLN/RMSNorm/RoPE/timestep/caption/connectors",
                                           residual_dtype="float32")
            check["torch"] = torch.__version__
            check["torch_cuda"] = torch.version.cuda
            check["compute_capability"] = cap
            check["loaded_memory"] = {"main": live.loaded_memory}
            check["shard_bytes"] = {"main": live.shard_bytes}
            check["cpu_memory"] = process_memory()
            check["patch_fingerprint"] = patch.fingerprint()
            check["wan_fingerprints"] = {"main": live.patch_fingerprint}
            print(json.dumps({"rank": rank, "GPU": torch.cuda.get_device_name(rank), "role": "ltx",
                              "geometry": live.geometry, "fp16_safe": ltx_options.fp16_safe, "residual": "FP32",
                              "attention_effective": live.attention.effective, "sequence_mode": config.sequence_mode,
                              "world_size": world, "weight_placement": config.weight_placement,
                              "shard_bytes": check["shard_bytes"], "load_s": round(check["load_s"], 2),
                              "load_timing": check["load_timing"]}, ensure_ascii=False, default=str), file=sys.stderr)
        elif not settings["probe_only"]:
            loras = role_options.get("loras", {})
            owners = {}
            swap = options.moe_residency == "swap" and len(experts) > 1
            # swap: загружаем в обратном порядке и паркуем всех, кроме последнего
            # (первого по sampling, обычно high) — пик VRAM = один эксперт.
            order = list(reversed(list(experts))) if swap else list(experts)
            for position, slot in enumerate(order):
                def make_backend(local_state, initial_pool=None, _slot=slot, _path=experts[slot]):
                    pool = initial_pool
                    if config.weight_placement == "ats" and pool is None:
                        from .ats_memory import ManagedShardPool
                        pool = ManagedShardPool(device)
                    return WanBackend(_path, config, device, options, settings.get("attention_policy"), _slot,
                                      tuple(loras.get(_slot, ())), pool, local_state)
                # Только первый эксперт получает уже созданный pool; остальные свой.
                owners[slot] = ResidentBackend(make_backend(None, managed_pool), make_backend, device)
                managed_pool = None
                if swap and position < len(order) - 1:
                    owners[slot].idle()
            owners = {slot: owners[slot] for slot in experts}
            from .wan_config import WanCheckpoint
            main_geometry = WanCheckpoint(next(iter(experts.values()))).model_config()
            main_dim = main_geometry["dim"]

            def uni3c_factory(path):
                def make(local_state, initial_pool=None):
                    from .wan_backend import Uni3CBackend
                    pool = initial_pool
                    if config.weight_placement == "ats" and pool is None:
                        from .ats_memory import ManagedShardPool
                        pool = ManagedShardPool(device)
                    return Uni3CBackend(path, config, device, options, settings.get("attention_policy"), main_dim,
                                        pool, local_state)
                return make
            def multitalk_factory(path):
                def make(local_state, initial_pool=None):
                    from .wan_backend import MultiTalkBackend
                    pool = initial_pool
                    if config.weight_placement == "ats" and pool is None:
                        from .ats_memory import ManagedShardPool
                        pool = ManagedShardPool(device)
                    return MultiTalkBackend(path, config, device, options, settings.get("attention_policy"), main_dim,
                                            main_geometry["num_layers"], pool, local_state)
                return make
            backend = WanExperts(owners, options.moe_residency, device, uni3c_factory, multitalk_factory)
            if swap:
                backend.active = order[-1]
            first = owners[order[-1]]
            check["role"] = "wan"
            check["quantization"] = {slot: (o._backend.quant_report if o._backend else "parked") for slot, o in owners.items()}
            check["experts"] = {slot: dict(identity=o._backend.identity if o._backend else "parked",
                                           lora=o._backend.lora_report if o._backend else "parked")
                                for slot, o in owners.items()}
            check["attention"] = first.attention.report()
            check["load_s"] = time.perf_counter() - start
            check["load_timing"] = {slot: (getattr(o._backend, "load_timing", None) if o._backend else "parked")
                                    for slot, o in owners.items()}
            check["config"] = config.to_dict()
            check["numeric_policy"] = dict(config.numeric_policy(None),
                                           compute_dtype=("FP16 scaled GEMM (Safe)" if options.fp16_safe else options.weight_dtype.upper() + " GEMM")
                                           + "; FP32 residual/modulation/LayerNorm/RMSNorm/RoPE/time embedding",
                                           residual_dtype="float32", activation_dtype="FP32 stream; FP16 GEMM/attention operands")
            check["torch"] = torch.__version__
            check["torch_cuda"] = torch.version.cuda
            from .telemetry import accelerator_inventory
            check["accelerators"] = accelerator_inventory()
            check["compute_capability"] = cap
            check["loaded_memory"] = {slot: (o._backend.loaded_memory if o._backend else "parked") for slot, o in owners.items()}
            check["shard_bytes"] = {slot: (o._backend.shard_bytes if o._backend else o._cached.parameter_bytes)
                                    for slot, o in owners.items()}
            check["cpu_memory"] = process_memory()
            check["patch_fingerprint"] = patch.fingerprint()
            check["wan_fingerprints"] = {slot: (o._backend.patch_fingerprint if o._backend else "parked")
                                         for slot, o in owners.items()}
            print(json.dumps({"rank": rank, "GPU": torch.cuda.get_device_name(rank), "capability": cap, "role": "wan",
                              "experts": list(owners), "residency": options.moe_residency,
                              "fp16_safe": options.fp16_safe, "residual": "FP32",
                              "attention_requested": config.requested_attention,
                              "attention_effective": first.attention.effective,
                              "sequence_mode": config.sequence_mode, "sequence_comm_dtype": config.sequence_comm_dtype,
                              "world_size": world, "rank_mapping": settings.get("selected_devices", []),
                              "weight_placement": config.weight_placement, "prefetch_blocks": config.prefetch_blocks,
                              "shard_bytes": check["shard_bytes"], "load_s": round(check["load_s"], 2),
                              "load_timing": check["load_timing"]}, ensure_ascii=False), file=sys.stderr)
        from .diagnostics import loaded_nccl
        check["loaded_nccl"] = loaded_nccl()
        check["uuid"] = str(getattr(torch.cuda.get_device_properties(rank), "uuid", "unknown"))
        check["rank_mapping"] = settings.get("selected_devices", [])
        check["world_size"] = world
        respond({"sequence": 0, "preflight": check})
        expected = 1
        stage_cache = {"dir": None, "tensors": {}}
        from .telemetry import RequestProgress
        progress = RequestProgress(Path(settings["report_dir"]) / f"{Path(settings['session_dir']).name}-rank{rank}-progress.json",
                                   rank=rank, pid=os.getpid(), uuid=settings["gpu_uuids"][rank])
        if backend is not None:
            backend.progress = progress
        profiled = {}
        for line in sys.stdin:
            req = json.loads(line)
            if req["command"] == "shutdown":
                break
            if req.get("sequence") != expected:
                raise RuntimeError("Нарушен порядковый номер команды")
            expected += 1
            if backend is None:
                raise RuntimeError("Сессия запущена только для диагностики")
            if req["command"] == "idle":
                stage_cache = {"dir": None, "tensors": {}}
                respond({"sequence": req["sequence"], "phase_offload": backend.idle()})
                continue
            if req["command"] == "end_run":
                report = backend.end_run()
                stage_cache = {"dir": None, "tensors": {}}
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                respond({"sequence": req["sequence"], "end_run": report})
                continue
            stage_dir = req.get("stage") or None
            progress.begin(req["sequence"], req["command"])
            request_start = time.perf_counter()
            if stage_dir != stage_cache["dir"]:
                stage_cache = {"dir": stage_dir, "tensors": {}}
            start = time.perf_counter()
            payload = read_payload(req["input"], device, base_directory=stage_dir,
                                   stage_cache=stage_cache["tensors"] if stage_dir else None)
            torch.cuda.synchronize(device)
            transfer_s = time.perf_counter() - start
            progress("input_transfer_s", transfer_s)
            profile_limit = max(1, int(os.environ.get("POWERSHARD_PROFILE_FORWARDS", "1")))
            if os.environ.get("POWERSHARD_PROFILE") == "1" and profiled.get(req["command"], 0) < profile_limit:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, profile_memory=True) as prof:
                    result, metrics = backend.call(req["command"], payload["args"], payload["kwargs"])
                trace = Path(settings["report_dir"]) / f"{Path(settings['session_dir']).name}-rank{rank}-seq{req['sequence']}.trace.json"
                prof.export_chrome_trace(str(trace))
                metrics["profiler_trace"] = str(trace)
                from .telemetry import profiler_summary
                metrics["kernels"] = profiler_summary(prof)
                profiled[req["command"]] = profiled.get(req["command"], 0) + 1
            else:
                result, metrics = backend.call(req["command"], payload["args"], payload["kwargs"])
            metrics["input_transfer_s"] = transfer_s
            progress("forward_s", metrics["forward_s"])
            progress("memory", metrics["memory"])
            progress("status", "FORWARD_COMPLETE")
            if rank == 0:
                print(json.dumps(dict(event="powershard_rpc", role="wan", sequence=req["sequence"], expert=metrics.get("expert"),
                                      forward_s=metrics["forward_s"], memory=metrics["memory"],
                                      attention_calls=metrics.get("attention", {}).get("call_counts"),
                                      sequence_communication=metrics.get("sequence_communication")), ensure_ascii=False),
                      file=sys.stderr)
            output_start = time.perf_counter()
            if rank == 0:
                write_payload(req["output"], result)
            metrics["output_serialization_s"] = time.perf_counter() - output_start
            del result, payload
            progress("status", "COMPLETE")
            metrics["progress_io"] = progress.report()
            metrics["worker_request_s"] = time.perf_counter() - request_start
            respond({"sequence": req["sequence"], "metrics": metrics})
    except BaseException:
        error = traceback.format_exc()
        if "progress" in locals():
            progress("error", error[-4000:])
            progress("status", "FAILED")
        print(error, file=sys.stderr)
        respond({"error": error})
        return 1
    finally:
        if "dist" in locals() and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
