"""FSDP2, meta-init и чтение только локальных строк каждого tensor."""
from contextlib import contextmanager
import hashlib
import json
import math
import time
import warnings
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy, CPUOffloadPolicy, OffloadPolicy
from torch.distributed.tensor import DTensor
from .checkpoint import Checkpoint
from .config import shard_bounds
from .operations import Operations, install_int8, dequantize_rows
from .attention import install_attention, install_sequence
from .patch_config import H3PatchConfig
from .fp16_safe import apply_fp16_safe


class Entrypoint(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, command, args, kwargs):
        if getattr(self.network, "_ps_safe_active", False):
            args, kwargs = list(args), dict(kwargs)
            if command == "preprocess_text":
                args[0] = args[0].float()
            elif command == "forward":
                args[0] = [x.float() for x in args[0]]
                if len(args) > 2:
                    args[2] = args[2].float()
                elif "context" in kwargs:
                    kwargs["context"] = kwargs["context"].float()
        # Всегда вызывается __call__ root. preprocess_text не обходит FSDP hooks.
        if command == "preprocess_text":
            return self.network.preprocess_text_embeds(*args, **kwargs)
        if command == "forward":
            return self.network(*args, **kwargs)
        raise ValueError(f"Неизвестный вызов H3: {command}")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory(device):
    if device.type != "cuda":
        return {"device": str(device), "cuda": "NOT_RUN"}
    free, total = torch.cuda.mem_get_info(device)
    return dict(device=str(device), allocated=torch.cuda.memory_allocated(device),
                reserved=torch.cuda.memory_reserved(device), peak_allocated=torch.cuda.max_memory_allocated(device),
                peak_reserved=torch.cuda.max_memory_reserved(device), free=free, total=total)


def wrap_fsdp(root, mesh, config=None):
    from .config import DistributedConfig
    config = config or DistributedConfig()
    policy = CPUOffloadPolicy(pin_memory=config.pin_memory) if config.cpu_offload else OffloadPolicy()
    units = []
    # Группы block включают нормы: native H3 читает некоторые norm weights напрямую.
    net = root.network
    for block in list(net.token_refiner.blocks) + list(net.blocks):
        if config.min_vram:
            # Parent owns norms/modulation, child owns its own weights. Child
            # __call__ hooks release attention before MLP all-gathers.
            for child in (block.attn, block.mlp):
                fully_shard(child, mesh=mesh, reshard_after_forward=True,
                    mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False), offload_policy=policy)
                units.append(child)
        fully_shard(block, mesh=mesh, reshard_after_forward=True,
                    mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False), offload_policy=policy)
        units.append(block)
    # Отдельные вспомогательные группы не растягивают root на весь проход.
    for name in ("condition_proj", "video_patch_proj", "audio_patch_proj", "final_layer", "time_embedder"):
        unit = getattr(net, name, None)
        if unit is not None:
            fully_shard(unit, mesh=mesh, reshard_after_forward=True,
                        mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False), offload_policy=policy)
            units.append(unit)
    fully_shard(root, mesh=mesh, reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False), offload_policy=policy)
    units.append(root)
    # Не prefetch: только текущая группа. Не хранить all-gather до backward.
    for unit in units:
        unit.set_modules_to_forward_prefetch([])
    # Отдельные цепочки: preprocess/refiner и denoising. Не prefetch неиспользуемые
    # группы root/condition в каждый denoising step. Максимум N следующих blocks.
    for chain in (list(net.token_refiner.blocks), list(net.blocks)):
        for i, unit in enumerate(chain):
            unit.set_modules_to_forward_prefetch(chain[i+1:i+1+config.prefetch_blocks])
    return units


def load_local(root, ckpt, device, rank, world, cpu_offload=False, generated_buffers=None, quant_map=None):
    from safetensors import safe_open
    consumed, evidence = set(), []
    aliases = list(root.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in aliases}) != len(aliases):
        raise ValueError("Обнаружены tied parameters: нужен отдельный audited mapping, загрузка остановлена")
    # При offload materialize CPU DTensor shards СРАЗУ; не full CPU models,
    # и не GPU shards с последующим ручным переносом вместо CPUOffloadPolicy.
    root.to_empty(device=torch.device("cpu") if cpu_offload else device)
    with torch.no_grad(), safe_open(str(ckpt.path), framework="pt", device="cpu") as f:
        for name, param in root.named_parameters():
            key = name.removeprefix("network.")
            source = ckpt.tensors.get(key)
            if source is None or tuple(source["shape"]) != tuple(param.shape):
                raise ValueError(f"Несовпадение checkpoint и модели: {key}")
            if not isinstance(param, DTensor) or tuple(p.dim for p in param.placements) != (0,):
                raise RuntimeError(f"Параметр не является FSDP Shard(0): {key}")
            a, b = shard_bounds(param.shape[0], rank, world)
            # safe_open.get_slice загружает только строки локального rank.
            piece = f.get_slice(key)[a:b]
            if piece.dtype == torch.int8 and param.dtype != torch.int8:
                # Квантованный embedding: параметр floating, источник I8.
                # Деквантизация локальных строк (ConvRot + row scale) — та же
                # математика, что Int8Linear.forward, но один раз при загрузке.
                # Веса Int8Linear (param.dtype == int8) остаются сырыми I8:
                # их деквантизирует forward по активным строкам.
                # quant_map приходит в двух формах: ключи-имена модулей из
                # ckpt.quantization() (install_int8) и полные имена тензоров
                # (probe); принимаем обе.
                qm = quant_map or {}
                conf = qm.get(key)
                if conf is None and key.endswith(".weight"):
                    conf = qm.get(key[:-len(".weight")])
                if conf is None:
                    raise ValueError(f"INT8-параметр без quant metadata: {key}")
                scale_key = key[:-len(".weight")] + ".weight_scale"
                scale_slice = f.get_slice(scale_key)[a:b]
                piece = dequantize_rows(piece, scale_slice, conf.get("convrot", False),
                                        conf.get("group_size", 256), dtype=param.dtype, check=True)
                consumed.add(scale_key)
            if piece.is_floating_point():
                if not torch.isfinite(piece).all():
                    raise FloatingPointError(f"Не конечные веса: {key}")
                if param.dtype == torch.float16 and piece.numel() and piece.float().abs().max() > 65504:
                    raise FloatingPointError(f"BF16->FP16 overflow: {key}, rows {a}:{b}")
            local = param.to_local()
            if local.shape != piece.shape:
                raise RuntimeError(f"FSDP padding/shape не совпал: {key}: {local.shape} != {piece.shape}")
            local.copy_(piece.to(dtype=local.dtype, device=local.device))
            evidence.append(dict(name=key, rows=[a,b], shape=list(param.shape), local_shape=list(local.shape),
                                 dtype=str(param.dtype), local_bytes=local.numel()*local.element_size(),
                                 device=str(local.device), pinned=local.is_pinned(),
                                 storage_bytes=local.untyped_storage().nbytes()))
            consumed.add(key)
            del piece
        for name, buf in list(root.named_buffers()):
            key = name.removeprefix("network.")
            if key in (generated_buffers or {}):
                src=generated_buffers[key]
            elif key in ckpt.tensors:
                src=f.get_tensor(key)
            else:
                raise ValueError(f"Неизвестный buffer: {key}")
            if src.shape != buf.shape or not torch.isfinite(src).all():
                raise ValueError(f"Неверный buffer: {key}")
            if buf.device != device:
                parent, _, leaf = name.rpartition(".")
                buf = buf.to(device)
                root.get_submodule(parent)._buffers[leaf] = buf
            buf.copy_(src.to(device=device, dtype=buf.dtype))
            consumed.add(key)
        unexpected = set(ckpt.tensors) - consumed
        if any(not k.endswith(".comfy_quant") for k in unexpected):
            raise ValueError(f"Непрочитанные веса: {sorted(unexpected)[:12]}")
    return evidence


def fsdp_groups(module):
    state = module._get_fsdp_state()
    if hasattr(state, "_fsdp_param_groups"):
        return state._fsdp_param_groups
    group = getattr(state, "_fsdp_param_group", None)
    return [] if group is None else [group]


def assert_sharded(root, cpu_offload=None):
    world = dist.get_world_size() if dist.is_initialized() else 1
    for name, p in root.named_parameters():
        if not isinstance(p, DTensor) or (world > 1 and p.to_local().numel() >= p.numel() and p.shape[0] >= world):
            raise RuntimeError(f"Полный параметр остался после forward: {name}")
        if p.requires_grad or p.grad is not None:
            raise RuntimeError(f"Ненужный gradient: {name}")
        if cpu_offload is not None:
            expected = "cpu" if cpu_offload else "cuda"
            if p.to_local().device.type != expected:
                raise RuntimeError(f"Shard {name} находится на {p.to_local().device}, ожидался {expected}")
    for module in root.modules():
        if isinstance(module, FSDPModule):
            for group in fsdp_groups(module):
                for p in group.fsdp_params:
                    if str(p.sharded_state).split(".")[-1] != "SHARDED":
                        raise RuntimeError("FSDP сохранил unsharded параметры после forward")


class H3Backend:
    def __init__(self, checkpoint, config, device, patch=None, attention_policy=None, role_options=None):
        from comfy.ldm.minimax.model import MiniMaxH3Model
        self.device, self.config = device, config
        self.patch = patch or H3PatchConfig()
        ckpt = Checkpoint(checkpoint)
        quant = ckpt.quantization()
        if bool(quant) != (config.precision == "int8_fp16"):
            raise ValueError("precision не соответствует формату checkpoint")
        self.identity = ckpt.identity()
        with torch.device("meta"):
            net = MiniMaxH3Model(**ckpt.model_config(), dtype=torch.float16, device="meta", operations=Operations)
            quant_map = install_int8(net, quant, config.dequant_rows) if quant else {}
            root = Entrypoint(net)
        root.eval().requires_grad_(False)
        from .attention_policy import AttentionDispatcher
        self.attention = AttentionDispatcher(config, attention_policy)
        install_attention(net, config, self.attention)
        from .attention import configure_safe_qk
        configure_safe_qk(net, ckpt)
        self.tracker = apply_fp16_safe(net, self.patch)
        from .memory_policy import apply_linear_policy
        apply_linear_policy(net, config)
        if config.backend == "fsdp2_sequence":
            install_sequence(net, config)
        self.world = dist.get_world_size()
        mesh = init_device_mesh(device.type, (self.world,), mesh_dim_names=("shard",))
        self.units = wrap_fsdp(root, mesh, config)
        self.shards = load_local(root, ckpt, device, dist.get_rank(), self.world, cpu_offload=config.cpu_offload, quant_map=quant_map)
        self.root = root
        sync(device)
        assert_sharded(root, config.cpu_offload if device.type == "cuda" else None)
        self.loaded_memory = memory(device)
        self.shard_bytes = sum(x["local_bytes"] for x in self.shards)
        if config.cpu_offload and config.pin_memory and any(not x["pinned"] for x in self.shards if x["local_bytes"]):
            raise RuntimeError("CPUOffloadPolicy(pin_memory=True) не создал pinned локальные shards")
        self.buffer_bytes = sum(b.numel()*b.element_size() for b in root.buffers())
        import math
        self.group_bytes = [sum(math.prod(p._orig_size)*p.sharded_param.element_size()
                               for group in fsdp_groups(unit) for p in group.fsdp_params) for unit in self.units]
        self.root_group_bytes = self.group_bytes[-1]
        from .telemetry import ForwardLedger
        self.ledger=ForwardLedger()
        names={id(m):n for n,m in root.named_modules()}
        for unit in self.units:
            self.ledger.attach(names.get(id(unit),"root"),unit,fsdp_groups(unit))
        self.memory_context={}
        for module in root.modules():
            if hasattr(module,"_ps_mlp_chunk"):
                module._ps_memory_context=self.memory_context
        from .spectrum import SpectrumEngine, install_spectrum
        from .spectrum_config import SpectrumConfig
        spectrum_config=SpectrumConfig(**(role_options or {}).get("spectrum",{}))
        self.spectrum_history_requested=spectrum_config.history_device
        if config.min_vram:
            from dataclasses import replace
            spectrum_config=replace(spectrum_config,history_device="cpu")
        self.spectrum=SpectrumEngine(spectrum_config,device)
        # После FSDP wrapping И загрузки: gate не является FSDP unit; ACTUAL вызывает
        # исходный FSDP child, FORECAST не вызывает его hooks/all-gather вообще.
        install_spectrum(net,self.spectrum)

    def plan_memory(self, command, args, kwargs=None):
        from .memory_policy import plan_forward
        if command=="forward":
            video,audio=args[0]
            context=args[2] if len(args)>2 else (kwargs or {})["context"]
            tokens=video.shape[2]*math.ceil(video.shape[3]/2)*math.ceil(video.shape[4]/2)+audio.shape[-1]+context.shape[1]
        else:
            tokens=args[0].shape[-2]
        hidden=self.root.network.hidden_size
        sequence=command=="forward" and self.config.backend=="fsdp2_sequence" and self.world>1
        local_tokens=math.ceil(tokens/(self.world if sequence else 1))
        activation=int((tokens*hidden*12+local_tokens*hidden*24))
        attn=self.root.network.blocks[0].attn
        # H3 inner=heads*head_dim НЕ равен hidden_size (7168 vs5376 Pruned).
        # Global gathered KV, cast/copy allowance и local packing; не NCCL bytes.
        # fp16-канал sequence (sequence_comm_dtype=fp16) вдвое меньше fp32.
        # Includes restored FP32 KV + compressed communication buffers. Halving
        # ALL workspace because only wire tensors are half was an underestimate.
        comm_factor = 8
        communication=int((2*math.ceil(tokens/self.world)*self.world+2*local_tokens)*attn.heads*attn.head_dim*comm_factor) if sequence else 0
        from .memory_policy import allocator_budget
        budget=allocator_budget(self.device)
        free=budget["planning_available_bytes"]
        plan=plan_forward(free,int(self.config.reserve_gib*2**30),activation,communication,
                          (self.root_group_bytes+sum(sorted(self.group_bytes[:-1],reverse=True)[:2])) if self.config.min_vram else max(self.group_bytes,default=0),
                          self.config.prefetch_blocks,self.config.memory_policy)
        from .memory_policy import apply_workspace_policy
        plan=apply_workspace_policy(plan,self.config)
        plan.update(allocator_budget=budget,free_at_boundary=budget["driver_free_bytes"],planning_available_bytes=free)
        payload=(kwargs or {}).get("minimax_payload",{})
        if payload.get("refs") or payload.get("keyframes"):
            plan["conditioning_tokens"]="UNKNOWN reference/keyframe expansion; token estimate is a lower bound"
        plan["estimated_global_tokens"]=int(tokens)
        plan["estimated_local_tokens"]=local_tokens
        if self.config.memory_policy=="auto" and dist.is_initialized():
            # Один согласованный бюджет на границе RPC; никаких collectives в MLP chunks.
            limit=torch.tensor([plan["effective_prefetch"],plan["mlp_budget_bytes"]],device=self.device,dtype=torch.int64)
            dist.all_reduce(limit,op=dist.ReduceOp.MIN)
            plan["effective_prefetch"],plan["mlp_budget_bytes"]=limit.cpu().tolist()
        if self.config.memory_policy=="auto":
            for chain in (self.root.network.token_refiner.blocks,self.root.network.blocks):
                for i,unit in enumerate(chain):
                    unit.set_modules_to_forward_prefetch(list(chain[i+1:i+1+plan["effective_prefetch"]]))
        self.memory_context.clear()
        self.memory_context.update(plan)
        return plan

    def call(self, command, args, kwargs):
        kwargs=dict(kwargs)
        metadata=kwargs.pop("_powershard_spectrum",None)
        if command=="forward":self.spectrum.begin(metadata)
        sync(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            free, _ = torch.cuda.mem_get_info(self.device)
            if free < self.config.reserve_gib * 2**30:
                if not getattr(self, "_reserve_warned", False):
                    warnings.warn("Свободная VRAM меньше выбранного резерва; прогноз не запрещает попытку inference")
                    self._reserve_warned = True
        start = time.perf_counter()
        plan=self.plan_memory(command,args,kwargs)
        self.ledger.reset()
        for module in self.root.modules():
            for key in ("_ps_mlp_report","_ps_shapes"):
                if hasattr(module,key):delattr(module,key)
        # Не наследуем inference_mode ComfyUI: FSDP hooks работают в no_grad.
        with torch.inference_mode(False), torch.no_grad():
            self.tracker.begin(self.device)
            try:
                from .telemetry import region
                with region("H3/"+command):result = self.root(command, args, kwargs)
                assert_sharded(self.root, self.config.cpu_offload if self.device.type == "cuda" else None)
            finally:
                for unit in self.units:
                    unit.reshard()
        sync(self.device)
        assert_sharded(self.root, self.config.cpu_offload if self.device.type == "cuda" else None)
        self.tracker.finish(result)
        sequence_state = getattr(self.root.network.blocks[0].attn,"_ps_sequence",{})
        sequence_active = command=="forward" and not self.spectrum.forecast and sequence_state.get("enabled",False)
        metrics = {"forward_s": time.perf_counter()-start, "memory": memory(self.device),
                        "sharded_after_forward": True, "backend": "fsdp2_sequence" if sequence_active else "fsdp2",
                        "requested_backend": self.config.backend,
                        "duplicated_compute": self.world > 1 and not sequence_active,
                        "world_size": self.world, "inter_gpu_sharding": self.world > 1,
                        "attention": self.attention.report()}
        metrics["dit_blocks_executed"]=(0 if self.spectrum.forecast else len(self.root.network.blocks)) if command=="forward" else 0
        from .topology import process_memory
        metrics["cpu_memory"] = process_memory()
        if command=="forward":
            metrics["spectrum"]=self.spectrum.report()
            metrics["spectrum"]["history_device_requested"]=self.spectrum_history_requested
        metrics["memory_plan"] = plan
        metrics["execution"] = self.ledger.report()
        metrics["mlp"] = {n:m._ps_mlp_report for n,m in self.root.named_modules() if hasattr(m,"_ps_mlp_report")}
        metrics["attention_shapes"] = {n:m._ps_shapes for n,m in self.root.named_modules() if hasattr(m,"_ps_shapes")}
        metrics["weight_memory"] = dict(persistent_shard_bytes=self.shard_bytes,
            cpu_shard_bytes=self.shard_bytes if self.config.cpu_offload else 0,
            gpu_shard_bytes=0 if self.config.cpu_offload else self.shard_bytes,
            replicated_buffer_bytes=self.buffer_bytes,
            largest_unsharded_group_bytes=max(self.group_bytes, default=0),
            prefetch_blocks=self.config.prefetch_blocks,
            active_plus_prefetch_parameter_bound_bytes=self.root_group_bytes+sum(sorted(self.group_bytes[:-1], reverse=True)[:(2 if self.config.min_vram else 1+self.config.prefetch_blocks)]),
            group_nesting="root + parent block + active attention/MLP" if self.config.min_vram else "root + block (+ prefetch)",
            bound_excludes="copy/communication buffers, temporary FP32 casts and dequant; inspect CUDA peak/trace",
            activation_workspace_bytes="NOT_SEPARATELY_ATTRIBUTED; included in CUDA allocated/peak")
        if sequence_active:
            total = self.root.network.blocks[0].attn._ps_sequence["total"]
            metrics["local_token_range"] = shard_bounds(total, dist.get_rank(), self.world)
            metrics["total_tokens"] = total
            metrics["sequence_communication"] = dict(kv_all_gathers=sequence_state.get("kv_collectives",0),
                gathered_bytes=sequence_state.get("kv_gathered_bytes",0),final_hidden_all_gathers=1,
                note="Gathered tensor bytes, not NCCL link traffic; K/V fused into one collective per block")
            ulysses=bool(sequence_state.get("ulysses"))
            metrics["sequence_communication"].update(requested_mode=sequence_state.get("requested_mode"),
                effective_mode=sequence_state.get("mode"),
                kv_all_gathers=0 if ulysses else sequence_state.get("kv_collectives",0),
                all_to_all_calls=3*len(self.root.network.blocks) if ulysses else 0,
                scale_all_reduces=sequence_state.get("scale_all_reduces",0)+(1 if ulysses and self.config.sequence_comm_dtype=="fp16" else 0),
                note="H3 token KV can use scaled FP16; Ulysses QKV use native dtype. Bytes are tensor sizes, not measured link traffic.")
        return result, metrics

    def idle(self):
        """Освобождение allocator один раз между фазами; CPU-шарды сохраняются."""
        if not self.config.cpu_offload:
            raise ValueError("Сохранение CPU-шардов требует CPUOffloadPolicy")
        before=memory(self.device)
        for unit in self.units:unit.reshard()
        assert_sharded(self.root,True)
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        return dict(before=before,after=memory(self.device),cpu_shard_bytes=self.shard_bytes,
                    retained="CPU shards + replicated buffers + CUDA/NCCL context")
