"""FSDP2 backend LTX-2 / 2.3 / 2.5 (и LTX-Video) в worker-процессе PowerShard.

Загрузка — общий загрузчик Wan (локальные строки Shard(0), bf16/fp8(_scaled) -> FP16 на GPU rank,
LoRA merge в FP32 по строкам, q/k bounds); FSDP units — transformer_blocks + вспомогательные модули;
команды: forward (генератор) и preprocess (текстовые коннекторы для host extra_conds / Duration head).
"""
import hashlib
import json
import math
import time
import warnings
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from .config import shard_bounds
from .fsdp_backend import assert_sharded, memory, sync, fsdp_groups
from .wan_backend import wrap_fsdp_units, load_wan_local


def ltx_patch_fingerprint(options, loras):
    value = dict(options=options.to_dict(), loras=loras, numeric="ltx-fp32-stream-v1")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class LTXBackend:
    def __init__(self, checkpoint, config, device, options, attention_policy=None, loras=(), managed_pool=None,
                 local_state=None):
        from .accel import FirstBlockCache
        from .attention_policy import AttentionDispatcher
        from .ltx_config import LTXCheckpoint, ltx_storage_ok
        from .ltx_model import (LTXEntrypoint, build_ltx_network, make_fp32_parameters, configure_ltx,
                                fsdp_unit_modules, qk_norm_names, apply_qk_scales)
        self.device, self.config, self.managed_pool, self.slot = device, config, managed_pool, "ltx"
        self.options = options
        ckpt = LTXCheckpoint(checkpoint)
        ltx_storage_ok(ckpt, config.precision)
        self.identity = dict(ckpt.identity(), storage=ckpt.storage()["kind"], slot="ltx")
        self.geometry = ckpt.model_config()
        net = build_ltx_network(self.geometry, torch.bfloat16 if options.weight_dtype == "bf16" else torch.float16)
        with torch.device("meta"):
            make_fp32_parameters(net)
            root = LTXEntrypoint(net)
        root.eval().requires_grad_(False)
        self.attention = AttentionDispatcher(config, attention_policy)
        self.tracker, self.sequence = configure_ltx(net, self.geometry, options, self.attention, config)
        from safetensors import safe_open
        from .quant_linear import install_quant_linears
        self.quant_report = install_quant_linears(
            net, ckpt.tensors, options.weight_format, options.compute,
            None if options.weight_format == "as_file" else ("transformer_blocks.",),
            torch.bfloat16 if options.weight_dtype == "bf16" else torch.float16, device,
            open_file=lambda table, key: safe_open(str(ckpt.path), framework="pt", device="cpu"))
        self.patch_fingerprint = ltx_patch_fingerprint(options, list(loras))
        net._ps_patch_fingerprint = self.patch_fingerprint
        self.spectrum = FirstBlockCache()   # ResidentBackend.end_run очищает его между задачами
        net._ps_block_cache = self.spectrum
        self.world = dist.get_world_size()
        mesh = init_device_mesh(device.type, (self.world,), mesh_dim_names=("shard",))
        self.units = wrap_fsdp_units(root, fsdp_unit_modules(net), list(net.transformer_blocks), mesh, config)
        merger = None
        if loras and local_state is None:
            from .wan_lora import LoraMerger
            merger = LoraMerger(loras, ckpt.tensors, label_prefix="LTX: ")
        self.lora_report = merger.report if merger else [dict(cached=True, **l) for l in loras]
        qk_names = qk_norm_names(net)
        # strict=False: полный checkpoint может содержать duration_head и т.п. (не части DiT).
        self.shards, maxima = load_wan_local(root, ckpt, device, dist.get_rank(), self.world, config.cpu_offload, {},
                                             managed_pool, local_state, merger, tuple(qk_names), {}, strict=False)
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
            raise ValueError("LTX CPU retention requires CPUOffloadPolicy")
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
        video = x[0] if isinstance(x, (list, tuple)) else x
        batch = video.shape[0]
        tokens = math.prod(video.shape[2:])
        net = self.root.network
        dim, heads = net.inner_dim, net.num_attention_heads
        head_dim = dim // heads
        sequence = self.world > 1 and shard_bounds(tokens, self.world - 1, self.world)[0] < tokens
        local = math.ceil(tokens / self.world) if sequence else tokens
        audio = x[1].shape[2] if isinstance(x, (list, tuple)) and len(x) > 1 else 0
        # residual + модулированный вход + q/k/v FP32 + attention + RoPE локальных строк + аудио поток.
        activation = int(batch * local * dim * 4 * 10 + batch * audio * 2048 * 4 * 12)
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
        chain = list(net.transformer_blocks)
        if self.config.memory_policy == "auto" and dist.is_initialized():
            limit = torch.tensor([plan["effective_prefetch"], plan["mlp_budget_bytes"]], device=self.device, dtype=torch.int64)
            dist.all_reduce(limit, op=dist.ReduceOp.MIN)
            local_prefetch = plan["effective_prefetch"]
            plan["effective_prefetch"], plan["mlp_budget_bytes"] = limit.cpu().tolist()
            if plan["effective_prefetch"] < local_prefetch:
                plan["prefetch_reason"] = "auto: a different rank had less estimated headroom (MIN consensus)"
            plan["prefetch_reduced"] = plan["effective_prefetch"] < plan["requested_prefetch"]
        # Каждый вызов заново: Block Cache мог отключить prefetch блока 0 в прошлом forward.
        for i, unit in enumerate(chain):
            unit.set_modules_to_forward_prefetch(chain[i + 1:i + 1 + plan["effective_prefetch"]])
        plan.update(estimated_global_tokens=int(tokens * batch), estimated_local_tokens=int(local * batch),
                    audio_tokens=int(audio * batch))
        self.memory_context.clear()
        self.memory_context.update(plan)
        if getattr(self, "progress", None):
            self.progress("memory_plan", plan)
            self.memory_context["_progress"] = self.progress
        return plan

    def call(self, command, args, kwargs):
        if command not in ("forward", "preprocess"):
            raise ValueError(f"LTX worker: неизвестная команда {command}")
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
        plan = self.plan_memory(args) if command == "forward" else {}
        self.ledger.reset()
        for module in self.root.modules():
            for key in ("_ps_mlp_report", "_ps_shapes"):
                if hasattr(module, key):
                    delattr(module, key)
        with torch.inference_mode(False), torch.no_grad():
            self.tracker.begin(self.device)
            try:
                from .telemetry import region
                with region("LTX/" + command):
                    result = self.root(command, args, kwargs)
                assert_sharded(self.root, self.config.cpu_offload if self.device.type == "cuda" else None)
            finally:
                for unit in self.units:
                    unit.reshard()
        sync(self.device)
        try:
            self.tracker.finish(result)
        except FloatingPointError as error:
            hint = "" if self.options.fp16_safe else (" Включите fp16_safe в PowerShard LTX Options (scaled FP16 GEMM) "
                                                       "или POWERSHARD_DEBUG_FINITE=1 для поиска модуля.")
            raise FloatingPointError(str(error).replace("H3 FP16 Safe", "LTX FP16") + hint) from None
        state = self.sequence
        net = self.root.network
        metrics = {"role": "ltx", "command": command, "expert": "ltx", "forward_s": time.perf_counter() - start,
                   "memory": memory(self.device), "sharded_after_forward": True, "backend": "fsdp2_sequence",
                   "world_size": self.world, "inter_gpu_sharding": self.world > 1,
                   "duplicated_compute": self.world > 1 and not state.get("enabled", False),
                   "attention": self.attention.report(), "checkpoint": self.identity, "lora": self.lora_report,
                   "entry_sync_s": entry_sync_s}
        if command == "forward":
            from .topology import process_memory
            metrics["cpu_memory"] = process_memory()
            metrics["memory_plan"] = plan
            metrics["execution"] = self.ledger.report()
            metrics["mlp"] = {n: m._ps_mlp_report for n, m in self.root.named_modules() if hasattr(m, "_ps_mlp_report")}
            metrics["block_cache"] = dict(self.spectrum.report(), skipped_this_call=bool(state.get("blocks_skipped")))
            metrics["dit_blocks"] = len(net.transformer_blocks)
            metrics["weight_memory"] = dict(persistent_shard_bytes=self.shard_bytes, placement=self.config.weight_placement,
                                            ats=self.managed_pool.report() if self.managed_pool else None,
                                            largest_unsharded_group_bytes=max(self.group_bytes, default=0),
                                            prefetch_blocks=plan["effective_prefetch"])
            if state.get("enabled"):
                metrics["total_tokens"] = state["total"]
                metrics["local_token_range"] = shard_bounds(state["total"], dist.get_rank(), self.world)
                metrics["sequence_mode_effective"] = "ulysses" if state["ulysses"] else "token"
                metrics["sequence_communication"] = dict(
                    mode=metrics["sequence_mode_effective"], communication_dtype=state.get("effective_comm_dtype"),
                    kv_collectives=state["kv_collectives"], output_all_to_all=state["out_collectives"],
                    v2a_all_reduces=state.get("v2a_collectives", 0), head_output_all_gathers=state["head_gathers"],
                    exchanged_bytes=state["kv_gathered_bytes"] + state["out_exchanged_bytes"],
                    guide_mask_tokens=state.get("guide_mask_tokens", 0),
                    note="Logical tensor bytes, not measured NCCL link traffic")
        return result, metrics
