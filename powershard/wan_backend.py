"""FSDP2 backend одного эксперта Wan и worker-владелец нескольких экспертов (MoE).

Повторно использует из PowerShard: Linear/Int8Linear/FP16 Safe GEMM, FiniteTracker,
AttentionDispatcher, memory planner, ShardBindings/ResidentBackend (RAM parking),
ForwardLedger, assert_sharded. Новое: загрузка Wan (префиксы ключей, bf16/fp8 ->
FP16 по локальным строкам, LoRA merge), FSDP units Wan, sequence-forward.
"""
from contextlib import nullcontext
import hashlib
import json
import math
import time
import warnings
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy, OffloadPolicy
from torch.distributed.tensor import DTensor
from .config import shard_bounds
from .fsdp_backend import assert_sharded, memory, sync, fsdp_groups
from .wan_config import WanCheckpoint, Uni3CCheckpoint, MultiTalkCheckpoint, model_kwargs, FP8_DTYPES

AUX_UNITS = ("patch_embedding", "text_embedding", "time_embedding", "time_projection", "img_emb", "ref_conv", "head")


def wan_patch_fingerprint(options, slot, loras):
    value = dict(options=options.to_dict(), slot=slot, loras=loras, numeric="wan-fp32-stream-v1")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def wrap_fsdp_units(root, unit_modules, chain, mesh, config):
    """Каждый unit — отдельная FSDP группа; root последним; prefetch только по цепочке блоков."""
    policy = CPUOffloadPolicy(pin_memory=config.pin_memory) if config.cpu_offload else OffloadPolicy()
    mp = MixedPrecisionPolicy(cast_forward_inputs=False)
    units = []
    for unit in unit_modules:
        fully_shard(unit, mesh=mesh, reshard_after_forward=True, mp_policy=mp, offload_policy=policy)
        units.append(unit)
    fully_shard(root, mesh=mesh, reshard_after_forward=True, mp_policy=mp, offload_policy=policy)
    units.append(root)
    for unit in units:
        unit.set_modules_to_forward_prefetch([])
    for i, unit in enumerate(chain):
        unit.set_modules_to_forward_prefetch(chain[i + 1:i + 1 + config.prefetch_blocks])
    return units


def wrap_wan_fsdp(root, mesh, config):
    from .wan_model import fsdp_unit_modules
    return wrap_fsdp_units(root, fsdp_unit_modules(root.network), list(root.network.blocks), mesh, config)


def load_wan_local(root, ckpt, device, rank, world, cpu_offload=False, quant_map=None, managed_pool=None,
                   local_state=None, lora=None, qk_names=(), generated_buffers=None, strict=True):
    """Локальные строки Shard(0) из checkpoint (или из RAM phase cache).

    Отличия от fsdp_backend.load_local: префикс ключей Wan, fp8(_scaled)->FP16
    деквантизация строк, LoRA merge в FP32 до записи, статистика q/k norm для
    attention scales. Одна MAX-коллективная операция на весь checkpoint.
    """
    from safetensors import safe_open
    from .operations import Linear, checkpoint_row_bounds, matmul_constants, dequantize_rows
    consumed, evidence = set(), []
    aliases = list(root.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in aliases}) != len(aliases):
        raise ValueError("Обнаружены tied parameters: загрузка остановлена")
    bounded = {(n + ".weight" if n else "weight"): m for n, m in root.named_modules()
               if isinstance(m, Linear) and getattr(m, "_ps_safe", False)
               and not getattr(m, "_ps_fp32", False) and m.weight.dtype == torch.float16}
    with managed_pool.scope() if managed_pool else nullcontext():
        root.to_empty(device=torch.device("cpu") if cpu_offload else device)
    if local_state is not None:
        params, buffers = dict(root.named_parameters()), dict(root.named_buffers())
        if params.keys() != local_state.parameters.keys() or buffers.keys() != local_state.buffers.keys():
            raise RuntimeError("Phase cache names differ from the reconstructed Wan model")
        with torch.no_grad():
            for name, param in params.items():
                if not isinstance(param, DTensor) or tuple(p.dim for p in param.placements) != (0,):
                    raise RuntimeError("Cached parameter is not FSDP Shard(0): " + name)
                local, piece = param.to_local(), local_state.parameters[name]
                if piece.device.type != "cpu" or local.shape != piece.shape or local.dtype != piece.dtype:
                    raise RuntimeError("Phase cache shape/dtype mismatch: " + name)
                local.copy_(piece)
                a, b = shard_bounds(param.shape[0], rank, world)
                evidence.append(dict(name=name.removeprefix("network."), rows=[a, b], shape=list(param.shape),
                                     local_shape=list(local.shape), dtype=str(param.dtype),
                                     local_bytes=local.numel() * local.element_size(), device=str(local.device),
                                     pinned=local.is_pinned(), source="RAM_LOCAL_SHARD_CACHE"))
            for name, buf in buffers.items():
                src = local_state.buffers[name]
                parent, _, leaf = name.rpartition(".")
                target = torch.empty_like(buf, device=device)
                target.copy_(src)
                root.get_submodule(parent)._buffers[leaf] = target
            if {n.rpartition(".")[0] for n in bounded} != local_state.weight_bounds.keys():
                raise RuntimeError("Phase cache FP16 Safe bounds differ")
            for name, bounds in local_state.weight_bounds.items():
                root.get_submodule(name)._ps_weight_bounds = bounds
            for name, scales in local_state.qk_scales.items():
                root.get_submodule(name)._ps_qk_scales = tuple(scales)
        if managed_pool:
            managed_pool.assert_parameters(root)
        return evidence, {}
    row_bounds, norm_maxima = {}, {}
    quant_map = quant_map or {}
    # Конвертация (bf16/fp8 -> fp16, LoRA, проверки) на GPU своего rank: на POWER9 эти проходы по
    # ~5 GB строк на CPU занимали большую часть загрузки. Чтение файла и итоговые shards — как раньше.
    work = device if device.type == "cuda" else torch.device("cpu")
    timing = dict(read_s=0., convert_s=0., lora_s=0., store_s=0., work_device=str(work))
    tick = time.perf_counter
    with torch.no_grad(), safe_open(str(ckpt.path), framework="pt", device="cpu") as f:
        for name, param in root.named_parameters():
            key = name.removeprefix("network.")
            if key.endswith(".qbytes"):
                # QuantLinear: упакованный QuantizedTensor (формат файла или квантование при загрузке) -> свои байты.
                from .quant_linear import build_bytes
                if not isinstance(param, DTensor) or tuple(p.dim for p in param.placements) != (0,):
                    raise RuntimeError(f"Параметр не является FSDP Shard(0): {key}")
                owner = root.get_submodule(name[:-len(".qbytes")])
                a, b = shard_bounds(param.shape[0], rank, world)
                started = tick()
                delta = (lambda k, shape, dev: lora.delta(k, 0, shape[0], shape, device=dev)) if lora else None
                full, used = build_bytes(owner, f, work, delta)
                local = param.to_local()
                if local.shape != full[a:b].shape:
                    raise RuntimeError(f"FSDP shape QuantLinear {key}: {local.shape} != {full[a:b].shape}")
                local.copy_(full[a:b])
                timing["convert_s"] += tick() - started
                consumed.update(used)
                evidence.append(dict(name=key, rows=[a, b], shape=list(param.shape), local_shape=list(local.shape),
                                     dtype="uint8", source_dtype="quantized:" + owner._ps_qplan.fmt,
                                     local_bytes=local.numel(), device=str(local.device), pinned=local.is_pinned()))
                del full
                continue
            source = ckpt.tensors.get(key)
            if source is None or tuple(source["shape"]) != tuple(param.shape):
                raise ValueError(f"Несовпадение Wan checkpoint и модели: {key} "
                                 f"({None if source is None else source['shape']} != {list(param.shape)})")
            if not isinstance(param, DTensor) or tuple(p.dim for p in param.placements) != (0,):
                raise RuntimeError(f"Параметр не является FSDP Shard(0): {key}")
            a, b = shard_bounds(param.shape[0], rank, world)
            offset = source.get("row_offset", 0)  # fused tensors (WanDancer in_proj -> q/k/v)
            quant_conf = source.get("comfy_quant")
            native_int8 = quant_conf is not None and quant_conf.get("format") == "int8_tensorwise" and (
                key in quant_map or key[:-len(".weight")] in quant_map)
            started = tick()
            if quant_conf is not None and not native_int8:
                # nvfp4 / mxfp8 / int4 / int8 / fp8 comfy_quant: весь тензор родным кодом ComfyUI -> свои строки.
                from .quant_formats import dequantize_module, module_entries
                module = key[:-len(".weight")]
                entries = module_entries(ckpt.tensors, module)
                full = dequantize_module(f, entries, module, quant_conf, list(param.shape), work,
                                         torch.bfloat16 if param.dtype == torch.bfloat16 else torch.float32)
                piece = full[offset + a:offset + b]
                consumed.update(entries)
                del full
                timing["read_s"] += tick() - started
                started = tick()
                source = dict(source, dtype="comfy_quant:" + str(quant_conf.get("format")))
            else:
                piece = f.get_slice(source["key"])[offset + a:offset + b]
                timing["read_s"] += tick() - started
                started = tick()
            if work.type == "cuda" and piece.dtype != torch.int8:
                piece = piece.to(work, non_blocking=False)
            if source["dtype"] in FP8_DTYPES:
                scale_key = ckpt.fp8_scale_key(key)
                piece = piece.float()
                if scale_key is not None:
                    scale = f.get_tensor(ckpt.tensors[scale_key]["key"]).float().to(piece.device)
                    if scale.numel() > 1:  # row-wise [out] / [out,1]
                        scale = scale.reshape(-1)[offset + a:offset + b].reshape((b - a,) + (1,) * (piece.ndim - 1))
                    piece = piece * scale.reshape(-1)[0] if scale.numel() == 1 else piece * scale
                    consumed.add(scale_key)
            elif piece.dtype == torch.int8 and param.dtype != torch.int8:
                conf = quant_map.get(key) or quant_map.get(key[:-len(".weight")] if key.endswith(".weight") else key)
                if conf is None:
                    raise ValueError(f"INT8-параметр без quant metadata: {key}")
                scale_key = key[:-len(".weight")] + ".weight_scale"
                piece = dequantize_rows(piece, f.get_slice(ckpt.tensors[scale_key]["key"])[a:b], conf.get("convrot", False),
                                        conf.get("group_size", 256), dtype=param.dtype, check=True)
                consumed.add(scale_key)
            timing["convert_s"] += tick() - started
            if lora:
                if param.dtype == torch.int8:
                    raise ValueError("LoRA нельзя слить в INT8 storage; используйте fp16/bf16/fp8 checkpoint")
                started = tick()
                delta = lora.delta(key, a, b, list(param.shape), device=piece.device)
                if delta is not None:
                    piece = piece.float() + delta
                timing["lora_s"] += tick() - started
            started = tick()
            if piece.is_floating_point():
                check = piece.float()
                if not torch.isfinite(check).all():
                    raise FloatingPointError(f"Не конечные веса: {key}")
                if param.dtype == torch.float16 and check.numel() and check.abs().max() > 65504:
                    raise FloatingPointError(f"Вес вне диапазона FP16: {key}, rows {a}:{b}")
                if key in qk_names:
                    norm_maxima[key] = float(check.abs().max()) if check.numel() else 0.
                del check
            local = param.to_local()
            if local.shape != piece.shape:
                raise RuntimeError(f"FSDP padding/shape не совпал: {key}: {local.shape} != {piece.shape}")
            if name in bounded:
                row_bounds[name] = (checkpoint_row_bounds(piece.to(torch.float16)) if piece.device.type == "cpu"
                                    else device_row_bounds(piece))
            local.copy_(piece.to(dtype=local.dtype), non_blocking=False)
            timing["store_s"] += tick() - started
            evidence.append(dict(name=key, rows=[a, b], shape=list(param.shape), local_shape=list(local.shape),
                                 dtype=str(param.dtype), source_dtype=source["dtype"],
                                 local_bytes=local.numel() * local.element_size(), device=str(local.device),
                                 pinned=local.is_pinned()))
            consumed.add(key)
            del piece
        for name, buf in list(root.named_buffers()):
            key = name.removeprefix("network.")
            if key in (generated_buffers or {}):
                src = generated_buffers[key]
            elif key in ckpt.tensors:
                src = f.get_tensor(ckpt.tensors[key]["key"])
            else:
                raise ValueError(f"Неизвестный buffer Wan: {key}")
            parent, _, leaf = name.rpartition(".")
            target = torch.empty_like(buf, device=device)
            target.copy_(src.to(dtype=buf.dtype))
            root.get_submodule(parent)._buffers[leaf] = target
            consumed.add(key)
        unexpected = sorted(k for k in set(ckpt.tensors) - consumed if not ckpt.is_metadata(k))
        if unexpected and strict:
            raise ValueError(f"Непрочитанные веса Wan checkpoint: {unexpected[:12]}")
    # Row-sharded weights: максимумы локальных строк дают точные глобальные bounds.
    names = sorted(row_bounds)
    norm_names = sorted(qk_names)
    values = [v for n in names for v in row_bounds[n]] + [norm_maxima.get(n, 0.) for n in norm_names]
    if values:
        maxima = torch.tensor(values, dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        values = maxima.cpu().tolist()
    for i, name in enumerate(names):
        module = bounded[name]
        module._ps_weight_bounds = matmul_constants(values[2 * i], values[2 * i + 1], module.in_features)
    norm_values = dict(zip(norm_names, values[2 * len(names):]))
    if managed_pool is not None:
        managed_pool.assert_parameters(root)
        for item in evidence:
            item["managed"] = True
    if evidence:
        evidence[0]["load_timing"] = timing  # первая запись: разбивка времени загрузки rank
    return evidence, norm_values


def device_row_bounds(piece):
    """checkpoint_row_bounds на GPU: max|w| и max_i sum_j |w_ij| по значениям, округлённым до FP16."""
    values = piece.to(torch.float16).float().abs()
    if values.ndim != 2:
        raise ValueError("Weight bounds require a two-dimensional checkpoint shard")
    if not values.numel():
        return 0., 0.
    return float(values.max()), float(values.sum(-1).max())


class WanBackend:
    """Один эксперт Wan (или единственная модель) на выбранном наборе GPU."""

    def __init__(self, checkpoint, config, device, options, attention_policy=None, slot="main", loras=(),
                 managed_pool=None, local_state=None):
        from .wan_model import (WanEntrypoint, build_network, make_fp32_parameters, configure_network,
                                qk_norm_names, apply_qk_scales, generated_buffers)
        from .operations import install_int8
        from .attention_policy import AttentionDispatcher
        self.device, self.config, self.managed_pool, self.slot = device, config, managed_pool, slot
        self.options = options
        ckpt = WanCheckpoint(checkpoint, options.model_type)
        # precision=int8_fp16: native int8 хранение (только int8_tensorwise). Иначе любые форматы деквантуются.
        native_int8 = config.precision == "int8_fp16" and options.weight_format == "dequantize"
        quant = ckpt.quantization() if native_int8 else {}
        if native_int8 and not quant:
            raise ValueError(f"precision=int8_fp16 требует int8_tensorwise checkpoint, а этот {ckpt.storage()['kind']}; "
                             "выберите precision=fp16")
        if quant and loras:
            raise ValueError("LoRA нельзя слить в INT8 checkpoint")
        self.identity = dict(ckpt.identity(), storage=ckpt.storage()["kind"], slot=slot)
        self.geometry = ckpt.model_config()
        weight_dtype = torch.bfloat16 if options.weight_dtype == "bf16" and not quant else torch.float16
        net = build_network(model_kwargs(self.geometry), dtype=weight_dtype)
        with torch.device("meta"):
            make_fp32_parameters(net)
            quant_map = install_int8(net, quant, config.dequant_rows) if quant else {}
            root = WanEntrypoint(net)
        root.eval().requires_grad_(False)
        self.attention = AttentionDispatcher(config, attention_policy)
        self.tracker, self.sequence = configure_network(net, config, options, self.attention)
        from safetensors import safe_open
        from .quant_linear import install_quant_linears
        self.quant_report = install_quant_linears(
            net, ckpt.tensors, options.weight_format, options.compute,
            None if options.weight_format == "as_file" else ("blocks.", "vace_blocks."), weight_dtype, device,
            open_file=lambda table, key: safe_open(str(ckpt.path), framework="pt", device="cpu"))
        self.patch_fingerprint = wan_patch_fingerprint(options, slot, list(loras))
        net._ps_patch_fingerprint = self.patch_fingerprint
        from .accel import FirstBlockCache
        self.spectrum = FirstBlockCache()   # PowerShard Block Cache; ResidentBackend.end_run очищает между задачами
        net._ps_block_cache = self.spectrum
        self.world = dist.get_world_size()
        mesh = init_device_mesh(device.type, (self.world,), mesh_dim_names=("shard",))
        self.units = wrap_wan_fsdp(root, mesh, config)
        merger = None
        if loras and local_state is None:
            from .wan_lora import LoraMerger
            merger = LoraMerger(loras, ckpt.tensors, label_prefix=f"{slot}: ")
        self.lora_report = merger.report if merger else [dict(cached=True, **l) for l in loras]
        qk_names = qk_norm_names(net)
        self.shards, maxima = load_wan_local(root, ckpt, device, dist.get_rank(), self.world, config.cpu_offload,
                                             quant_map, managed_pool, local_state, merger, tuple(qk_names),
                                             generated_buffers(root))
        if local_state is None:
            apply_qk_scales(net, maxima)
        self.load_timing = next((x["load_timing"] for x in self.shards if "load_timing" in x), None)
        self.root = root
        sync(device)
        assert_sharded(root, config.cpu_offload if device.type == "cuda" else None)
        self.loaded_memory = memory(device)
        self.shard_bytes = sum(x["local_bytes"] for x in self.shards)
        if config.cpu_offload and config.pin_memory and any(not x["pinned"] for x in self.shards if x["local_bytes"]):
            raise RuntimeError("CPUOffloadPolicy(pin_memory=True) не создал pinned локальные shards")
        self.buffer_bytes = sum(b.numel() * b.element_size() for b in root.buffers())
        self.group_bytes = [sum(math.prod(p._orig_size) * p.sharded_param.element_size()
                                for group in fsdp_groups(unit) for p in group.fsdp_params) for unit in self.units]
        self.root_group_bytes = self.group_bytes[-1]
        from .phase_cache import ShardBindings
        self.state_bindings = ShardBindings(root)
        from .telemetry import ForwardLedger
        self.ledger = ForwardLedger()
        names = {id(m): n for n, m in root.named_modules()}
        for unit in self.units:
            self.ledger.attach(names.get(id(unit), "root"), unit, fsdp_groups(unit))
        self.memory_context = {}
        for module in net.modules():
            if hasattr(module, "_ps_mlp_policy"):
                module._ps_memory_context = self.memory_context

    def idle(self):
        if not self.config.cpu_offload:
            raise ValueError("Wan CPU retention requires CPUOffloadPolicy")
        before = memory(self.device)
        for unit in self.units:
            unit.reshard()
        assert_sharded(self.root, True)
        sync(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return dict(before=before, after=memory(self.device), cpu_shard_bytes=self.shard_bytes,
                    retained="CPU shards + small CUDA buffers/NCCL context")

    def plan_memory(self, args):
        from .memory_policy import plan_forward, available_memory
        x = args[0]
        context = args[2] if len(args) > 2 and isinstance(args[2], torch.Tensor) else None
        batch = x.shape[0]
        tokens = x.shape[2] * math.ceil(x.shape[3] / 2) * math.ceil(x.shape[4] / 2)
        net = self.root.network
        dim, heads, head_dim = net.dim, net.num_heads, net.dim // net.num_heads
        sequence = self.world > 1 and shard_bounds(tokens, self.world - 1, self.world)[0] < tokens
        local = math.ceil(tokens / self.world) if sequence else tokens
        # residual, modulated input, q/k/v FP32, attention output, cross/residual temporaries.
        activation = int(batch * local * dim * 4 * 8 + (0 if context is None else context.numel() * 4 * 4))
        comm_bytes = 2 if self.config.sequence_comm_dtype == "fp16" else 4
        if sequence and self.config.sequence_mode == "ulysses":
            padded = math.ceil(heads / self.world) * self.world
            communication = 12 * batch * math.ceil(tokens / self.world) * padded * head_dim * comm_bytes
        elif sequence:
            communication = batch * (2 * tokens + 2 * local) * dim * comm_bytes + batch * 2 * tokens * dim * 2
        else:
            communication = 0
        snapshot = available_memory(self.device)
        plan = plan_forward(snapshot["free"], int(self.config.reserve_gib * 2**30), activation, int(communication),
                            max(self.group_bytes[:-1], default=0), self.config.prefetch_blocks, self.config.memory_policy,
                            reusable_bytes=snapshot["reusable_cache_bytes"])
        plan["allocator"] = snapshot["allocator"]
        if self.config.memory_policy == "auto" and dist.is_initialized():
            limit = torch.tensor([plan["effective_prefetch"], plan["mlp_budget_bytes"]], device=self.device, dtype=torch.int64)
            dist.all_reduce(limit, op=dist.ReduceOp.MIN)
            local_prefetch = plan["effective_prefetch"]
            plan["effective_prefetch"], plan["mlp_budget_bytes"] = limit.cpu().tolist()
            if plan["effective_prefetch"] < local_prefetch:
                plan["prefetch_reason"] = "auto: a different rank had less estimated headroom (MIN consensus)"
            plan["prefetch_reduced"] = plan["effective_prefetch"] < plan["requested_prefetch"]
        # Каждый вызов заново: Block Cache мог снять prefetch блока 0 в прошлом forward.
        chain = list(net.blocks)
        for i, unit in enumerate(chain):
            unit.set_modules_to_forward_prefetch(chain[i + 1:i + 1 + plan["effective_prefetch"]])
        plan.update(estimated_global_tokens=int(tokens * batch), estimated_local_tokens=int(local * batch))
        self.memory_context.clear()
        self.memory_context.update(plan)
        if getattr(self, "progress", None):
            self.progress("memory_plan", plan)
            self.memory_context["_progress"] = self.progress
        signature = (plan["requested_prefetch"], plan["effective_prefetch"], plan["policy"], plan["prefetch_reason"])
        if getattr(self, "_last_prefetch_plan", None) != signature:
            if dist.get_rank() == 0:
                import sys
                print(json.dumps(dict(event="powershard_prefetch", role="wan", slot=self.slot, requested=signature[0],
                                      effective=signature[1], policy=signature[2], reason=signature[3])), file=sys.stderr)
            self._last_prefetch_plan = signature
        return plan

    def call(self, command, args, kwargs):
        if command != "forward":
            raise ValueError(f"Wan worker: неизвестная команда {command}")
        entry = time.perf_counter()
        sync(self.device)
        entry_sync_s = time.perf_counter() - entry
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            free, _ = torch.cuda.mem_get_info(self.device)
            if free < self.config.reserve_gib * 2**30 and not getattr(self, "_reserve_warned", False):
                warnings.warn("Свободная VRAM меньше выбранного резерва; прогноз не запрещает попытку inference")
                self._reserve_warned = True
        start = time.perf_counter()
        plan = self.plan_memory(args)
        plan_s = time.perf_counter() - start
        self.ledger.reset()
        for module in self.root.modules():
            for key in ("_ps_mlp_report", "_ps_shapes"):
                if hasattr(module, key):
                    delattr(module, key)
        with torch.inference_mode(False), torch.no_grad():
            self.tracker.begin(self.device)
            compute_start = time.perf_counter()
            try:
                from .telemetry import region
                with region("Wan/" + self.slot):
                    result = self.root(command, args, kwargs)
                assert_sharded(self.root, self.config.cpu_offload if self.device.type == "cuda" else None)
            finally:
                for unit in self.units:
                    unit.reshard()
        compute_enqueue_s = time.perf_counter() - compute_start
        wait_start = time.perf_counter()
        sync(self.device)
        finish_wait_s = time.perf_counter() - wait_start
        finite_start = time.perf_counter()
        try:
            self.tracker.finish(result)
        except FloatingPointError as error:
            hint = "" if self.options.fp16_safe else (" Включите fp16_safe в PowerShard Wan Options (scaled FP16 GEMM) "
                                                       "или POWERSHARD_DEBUG_FINITE=1 для поиска модуля.")
            raise FloatingPointError(str(error).replace("H3 FP16 Safe", "Wan FP16") + hint) from None
        finite_s = time.perf_counter() - finite_start
        state = self.sequence
        net = self.root.network
        metrics = {"role": "wan", "expert": self.slot, "forward_s": time.perf_counter() - start, "memory": memory(self.device),
                   "sharded_after_forward": True, "backend": "fsdp2_sequence", "requested_backend": self.config.backend,
                   "duplicated_compute": self.world > 1 and not state.get("enabled", False),
                   "world_size": self.world, "inter_gpu_sharding": self.world > 1,
                   "attention": self.attention.report(), "checkpoint": self.identity, "lora": self.lora_report}
        metrics["wall_phases"] = dict(entry_sync_s=entry_sync_s, memory_plan_s=plan_s,
                                      compute_enqueue_and_reshard_s=compute_enqueue_s, finish_cuda_wait_s=finish_wait_s,
                                      finite_validation_s=finite_s,
                                      note="Enqueue includes blocking collectives/CPU work; phases are not exclusive GPU times.")
        metrics["dit_blocks_executed"] = 1 if state.get("blocks_skipped") else len(net.blocks)
        metrics["block_cache"] = dict(self.spectrum.report(), skipped_this_call=bool(state.get("blocks_skipped")))
        from .topology import process_memory
        metrics["cpu_memory"] = process_memory()
        metrics["memory_plan"] = plan
        metrics["execution"] = self.ledger.report()
        metrics["mlp"] = {n: m._ps_mlp_report for n, m in self.root.named_modules() if hasattr(m, "_ps_mlp_report")}
        metrics["weight_memory"] = dict(persistent_shard_bytes=self.shard_bytes,
                                        cpu_shard_bytes=self.shard_bytes if self.config.cpu_offload else 0,
                                        gpu_shard_bytes=self.shard_bytes if self.config.weight_placement == "gpu" else 0,
                                        managed_shard_bytes=self.shard_bytes if self.managed_pool else 0,
                                        placement=self.config.weight_placement,
                                        ats=self.managed_pool.report() if self.managed_pool else None,
                                        replicated_buffer_bytes=self.buffer_bytes,
                                        largest_unsharded_group_bytes=max(self.group_bytes, default=0),
                                        prefetch_blocks=plan["effective_prefetch"])
        if state.get("enabled"):
            metrics["total_tokens"] = state["total"]
            metrics["local_token_range"] = shard_bounds(state["total"], dist.get_rank(), self.world)
            metrics["sequence_mode_effective"] = "ulysses" if state["ulysses"] else "token"
            metrics["padded_heads"] = state.get("padded_heads", net.num_heads)
            metrics["sequence_communication"] = dict(
                mode=metrics["sequence_mode_effective"], communication_dtype=state.get("effective_comm_dtype"),
                kv_all_gathers=state["kv_collectives"] if not state["ulysses"] else 0,
                qkv_all_to_all=state["kv_collectives"] if state["ulysses"] else 0,
                output_all_to_all=state["out_collectives"], scale_all_reduces=state["scale_collectives"],
                exchanged_bytes=state["kv_gathered_bytes"] + state["out_exchanged_bytes"],
                head_output_all_gathers=state["head_gathers"],
                note="Logical tensor bytes, not measured NCCL link traffic")
        return result, metrics


class Uni3CBackend:
    """Uni3C ControlNet для Wan: собственный FSDP root в том же process group.

    Загружается лениво при первом вызове с Uni3C (все rank получают тот же RPC),
    паркуется в RAM вместе с экспертами. Блоки получают те же локальные строки,
    что основная модель; self-attention распределена так же (token/Ulysses).
    """

    def __init__(self, path, config, device, options, attention_policy, main_dim, managed_pool=None, local_state=None):
        import comfy.ldm.wan.uni3c as uni3c
        from .attention_policy import AttentionDispatcher
        from .fp16_safe import FiniteTracker, install_safe_operations
        from .wan_model import (WanOperations, Uni3CEntrypoint, install_wan_compute, qk_norm_names, apply_qk_scales,
                                generated_buffers)
        self.device, self.config, self.managed_pool, self.slot = device, config, managed_pool, "uni3c"
        ckpt = Uni3CCheckpoint(path)
        geometry = ckpt.model_config()
        if geometry["out_proj_dim"] != main_dim:
            raise ValueError(f"Uni3C выдаёт residual {geometry['out_proj_dim']}, а Wan dim={main_dim} (нужен Wan 14B)")
        self.identity = dict(ckpt.identity(), geometry=geometry)
        with torch.device("meta"):
            net = uni3c.WanUni3CControlnet(**geometry, device="meta", dtype=torch.float16, operations=WanOperations)
            root = Uni3CEntrypoint(net)
        root.eval().requires_grad_(False)
        self.attention = AttentionDispatcher(config, attention_policy)
        install_safe_operations(net, FiniteTracker(False), options.fp16_safe)
        self.state = install_wan_compute(net, self.attention, config, options)
        self.world = dist.get_world_size()
        mesh = init_device_mesh(device.type, (self.world,), mesh_dim_names=("shard",))
        units = list(net.controlnet_blocks) + list(net.proj_out)
        for name in ("controlnet_patch_embedding", "controlnet_mask_embedding", "proj_in"):
            module = getattr(net, name, None)
            if module is not None and any(True for _ in module.parameters()):
                units.append(module)
        # Блоки Uni3C вызываются отдельными вызовами root: без cross-call prefetch.
        self.units = wrap_fsdp_units(root, units, [], mesh, config)
        self.shards, maxima = load_wan_local(root, ckpt, device, dist.get_rank(), self.world, config.cpu_offload, {},
                                             managed_pool, local_state, None, tuple(qk_norm_names(net)),
                                             generated_buffers(root), strict=False)
        if local_state is None:
            apply_qk_scales(net, maxima)
        self.root = root
        sync(device)
        assert_sharded(root, config.cpu_offload if device.type == "cuda" else None)
        self.shard_bytes = sum(x["local_bytes"] for x in self.shards)
        from .phase_cache import ShardBindings
        self.state_bindings = ShardBindings(root)

    def idle(self):
        if not self.config.cpu_offload:
            raise ValueError("Uni3C CPU retention requires CPUOffloadPolicy")
        for unit in self.units:
            unit.reshard()
        sync(self.device)
        return dict(after=memory(self.device), cpu_shard_bytes=self.shard_bytes)


class MultiTalkBackend:
    """InfiniteTalk/MultiTalk model patch: 40 audio cross-attention блоков, своя FSDP root.

    Вызывается изнутри блока генератора (после cross-attention, как native attn2_patch): по одному
    вызову root на блок, одинаково на всех rank. audio_proj остаётся на host.
    """

    def __init__(self, path, config, device, options, attention_policy, main_dim, main_layers, managed_pool=None,
                 local_state=None):
        from .attention_policy import AttentionDispatcher
        from .fp16_safe import FiniteTracker, install_safe_operations
        from .wan_model import WanOperations
        from .wan_extras import MultiTalkBlocks, MultiTalkEntrypoint, install_multitalk_compute
        self.device, self.config, self.managed_pool, self.slot = device, config, managed_pool, "multitalk"
        ckpt = MultiTalkCheckpoint(path)
        geometry = ckpt.model_config()
        if geometry["in_dim"] != main_dim or geometry["num_layers"] != main_layers:
            raise ValueError(f"InfiniteTalk patch dim={geometry['in_dim']}/{geometry['num_layers']} блоков, а Wan "
                             f"dim={main_dim}/{main_layers} блоков (нужен Wan 2.1 14B)")
        self.identity = dict(ckpt.identity(), geometry=geometry)
        with torch.device("meta"):
            net = MultiTalkBlocks(geometry["in_dim"], geometry["out_dim"], geometry["num_layers"], dtype=torch.float16,
                                  device="meta", operations=WanOperations)
            root = MultiTalkEntrypoint(net)
        root.eval().requires_grad_(False)
        self.attention = AttentionDispatcher(config, attention_policy)
        install_safe_operations(net, FiniteTracker(False), options.fp16_safe)
        install_multitalk_compute(net, self.attention, options)
        self.world = dist.get_world_size()
        mesh = init_device_mesh(device.type, (self.world,), mesh_dim_names=("shard",))
        self.units = wrap_fsdp_units(root, list(net.blocks), [], mesh, config)
        self.shards, _ = load_wan_local(root, ckpt, device, dist.get_rank(), self.world, config.cpu_offload, {},
                                        managed_pool, local_state, None, (), {}, strict=True)
        self.root = root
        sync(device)
        assert_sharded(root, config.cpu_offload if device.type == "cuda" else None)
        self.shard_bytes = sum(x["local_bytes"] for x in self.shards)
        from .phase_cache import ShardBindings
        self.state_bindings = ShardBindings(root)

    def idle(self):
        if not self.config.cpu_offload:
            raise ValueError("InfiniteTalk CPU retention requires CPUOffloadPolicy")
        for unit in self.units:
            unit.reshard()
        sync(self.device)
        return dict(after=memory(self.device), cpu_shard_bytes=self.shard_bytes)


class WanExperts:
    """Владелец экспертов в одном worker: общий process group, отдельные FSDP roots.

    residency=both: оба эксперта активны (VRAM при placement=gpu).
    residency=swap: при смене эксперта неактивный паркуется в RAM (ResidentBackend.idle),
    один раз за переход high->low, а не на каждом шаге.
    Uni3C ControlNet (если используется) — ещё один ResidentBackend.
    """

    def __init__(self, owners, residency, device, uni3c_factory=None, multitalk_factory=None):
        self.owners, self.residency, self.device = owners, residency, device
        self.active = None
        self._progress = None
        self.uni3c_factory = uni3c_factory
        self.uni3c_key = None
        self.uni3c_owner = None
        self.multitalk_factory = multitalk_factory
        self.multitalk_key = None
        self.multitalk_owner = None

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value):
        self._progress = value
        for owner in self.owners.values():
            owner.progress = value

    def __getattr__(self, name):
        # attention/identity и т.п. для preflight берутся у активного (живого) эксперта.
        owners = self.__dict__.get("owners")
        if not owners:
            raise AttributeError(name)
        active = self.__dict__.get("active")
        live = [o for o in owners.values() if o.__dict__.get("_backend") is not None]
        owner = owners.get(active) if active in owners and owners[active].__dict__.get("_backend") is not None else (live or [None])[0]
        if owner is None:
            raise AttributeError(name)
        return getattr(owner, name)

    def uni3c_runtime(self, spec):
        import gc
        from .phase_cache import ResidentBackend
        key = (spec["path"], spec.get("size"), spec.get("mtime_ns"))
        if self.uni3c_key != key:
            self.uni3c_owner = None
            gc.collect()
            factory = self.uni3c_factory(spec["path"])
            self.uni3c_owner = ResidentBackend(factory(None), factory, self.device)
            self.uni3c_key = key
        owner = self.uni3c_owner
        if owner._backend is None:  # RAM-parked между задачами
            owner._backend = owner._factory(owner._cached)
            owner._cached = None
        return owner._backend

    def multitalk_runtime(self, spec):
        import gc
        from .phase_cache import ResidentBackend
        key = (spec["path"], spec.get("size"), spec.get("mtime_ns"))
        if self.multitalk_key != key:
            self.multitalk_owner = None
            gc.collect()
            factory = self.multitalk_factory(spec["path"])
            self.multitalk_owner = ResidentBackend(factory(None), factory, self.device)
            self.multitalk_key = key
        owner = self.multitalk_owner
        if owner._backend is None:
            owner._backend = owner._factory(owner._cached)
            owner._cached = None
        return owner._backend

    def call(self, command, args, kwargs):
        kwargs = dict(kwargs)
        slot = kwargs.pop("_powershard_expert", None) or next(iter(self.owners))
        if slot not in self.owners:
            raise ValueError(f"Wan expert {slot!r} не загружен в этой session: {sorted(self.owners)}")
        parked = {}
        switch_start = time.perf_counter()
        if self.residency == "swap" and self.active not in (None, slot):
            parked[self.active] = self.owners[self.active].idle()
        switch_s = time.perf_counter() - switch_start
        self.active = slot
        if kwargs.get("_powershard_uni3c"):
            if self.uni3c_factory is None:
                raise RuntimeError("Uni3C factory не настроен в worker")
            kwargs["_powershard_uni3c_runtime"] = self.uni3c_runtime(kwargs["_powershard_uni3c"])
        if kwargs.get("_powershard_multitalk"):
            if self.multitalk_factory is None:
                raise RuntimeError("InfiniteTalk factory не настроен в worker")
            kwargs["_powershard_multitalk_runtime"] = self.multitalk_runtime(kwargs["_powershard_multitalk"])
        result, metrics = self.owners[slot].call(command, args, kwargs)
        metrics["expert_switch"] = dict(residency=self.residency, parked=parked, park_s=switch_s)
        if self.uni3c_owner is not None and self.uni3c_owner._backend is not None:
            metrics["uni3c"] = dict(identity=self.uni3c_owner._backend.identity,
                                    attention=self.uni3c_owner._backend.attention.report().get("effective_used"))
        if self.multitalk_owner is not None and self.multitalk_owner._backend is not None:
            metrics["multitalk"] = dict(identity=self.multitalk_owner._backend.identity)
        return result, metrics

    def end_run(self):
        # Кэш pose branch Animate2 живёт в пределах одной задачи (как native PoseBranchCache.free на cleanup).
        for owner in self.owners.values():
            backend = owner.__dict__.get("_backend")
            if backend is not None:
                backend.root.network.__dict__.pop("_ps_pose_cache", None)
        return {slot: owner.end_run() for slot, owner in self.owners.items()}

    def idle(self):
        reports = {slot: owner.idle() for slot, owner in self.owners.items()}
        if self.uni3c_owner is not None:
            reports["uni3c"] = self.uni3c_owner.idle()
        if self.multitalk_owner is not None:
            reports["multitalk"] = self.multitalk_owner.idle()
        self.active = None
        return dict(after=memory(self.device), experts=reports,
                    cpu_shard_bytes=sum(r.get("cpu_shard_bytes", 0) for r in reports.values()))
