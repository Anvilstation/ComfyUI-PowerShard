"""Encoder role существующего FSDP2 runtime, один process group на выбранный набор."""
import inspect
import math
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy, OffloadPolicy
from torch.distributed.device_mesh import init_device_mesh
from .qwen import QwenOperations, QwenEntrypoint, install_qwen_compute, infer_qwen_config, QwenConfig
from .checkpoint import Checkpoint
from .operations import install_int8
from .fsdp_backend import load_local, assert_sharded, memory, sync, fsdp_groups
from .telemetry import ForwardLedger


class QwenBackend:
    def __init__(self,checkpoint,config,device,patch=None,attention_policy=None,role_options=None,managed_pool=None,local_state=None):
        from comfy.text_encoders.minimax import MiniMaxH3TEModel
        from comfy.text_encoders.llama import Qwen3VL_32BConfig
        from .attention_policy import AttentionDispatcher
        self.config,self.device,self.managed_pool=config,device,managed_pool
        self.world=dist.get_world_size()
        ckpt=Checkpoint(checkpoint);self.identity=ckpt.identity();geometry=infer_qwen_config(ckpt.tensors)
        quant=ckpt.quantization()
        if bool(quant)!=(config.precision=="int8_fp16"):
            raise ValueError("Qwen precision не соответствует реальному checkpoint storage")
        supported=inspect.signature(Qwen3VL_32BConfig).parameters
        model_config={k:v for k,v in geometry.items() if k in supported}
        with torch.device("meta"):
            encoder=MiniMaxH3TEModel(device="meta",dtype=torch.float16,model_options={
                "custom_operations":QwenOperations,"qwen3vl_32b_model_config":model_config})
            root=QwenEntrypoint(encoder)
            if root.network.model.layers[0].self_attn.head_dim!=geometry["head_dim"]:
                raise ValueError("Native Qwen config не представляет head_dim checkpoint")
            quant_map=install_int8(root.network,quant,config.dequant_rows) if quant else {}
        # Скаляр native SDClip wrapper не входит в encoder checkpoint и не нужен FSDP.
        encoder.qwen3vl_32b.logit_scale=torch.nn.Parameter(torch.tensor(4.6055),requires_grad=False)
        root.eval().requires_grad_(False);root._ps_device=device
        self.attention=AttentionDispatcher(config,attention_policy)
        cached_qk={name.removeprefix("network."):scales for name,scales in local_state.qk_scales.items()} if local_state else None
        self.tracker=install_qwen_compute(root.network,self.attention,ckpt,config.backend=="fsdp2_sequence",QwenConfig(**(role_options or {})),cached_qk)
        root.network._ps_patch_fingerprint=patch.fingerprint() if patch else "qwen-native-safe"
        mesh=init_device_mesh(device.type,(self.world,),mesh_dim_names=("shard",))
        policy=CPUOffloadPolicy(pin_memory=config.pin_memory) if config.cpu_offload else OffloadPolicy()
        self.units=[]
        def wrap(unit):
            fully_shard(unit,mesh=mesh,reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False),offload_policy=policy)
            unit.set_modules_to_forward_prefetch([])
            self.units.append(unit)
        for layer in root.network.model.layers:wrap(layer)
        wrap(root.network.model.embed_tokens)
        visual=root.network.visual
        for layer in visual.blocks:wrap(layer)
        for unit in (visual.patch_embed,visual.merger,*visual.deepstack_merger_list):wrap(unit)
        # pos_embed.weight читается напрямую в native interpolation: принадлежит
        # visual root, materialized ДО fast_pos_embed_interpolate(), не отдельному hook.
        wrap(visual)
        wrap(root)
        for chain in (root.network.model.layers,visual.blocks):
            for i,unit in enumerate(chain):
                unit.set_modules_to_forward_prefetch(list(chain[i+1:i+1+config.prefetch_blocks]))
        dim=visual.rotary_pos_emb.dim
        generated={"visual.rotary_pos_emb.inv_freq":1.0/(10000.**(torch.arange(0,dim,2,dtype=torch.float32)/dim))}
        self.shards=load_local(root,ckpt,device,dist.get_rank(),self.world,config.cpu_offload,generated,quant_map,managed_pool,local_state)
        self.root=root;self.shard_bytes=sum(x["local_bytes"] for x in self.shards)
        if config.cpu_offload and config.pin_memory and any(not s["pinned"] for s in self.shards if s["local_bytes"]):
            raise RuntimeError("Qwen CPUOffloadPolicy не создал pinned local shards")
        self.group_bytes=[sum(math.prod(p._orig_size)*p.sharded_param.element_size() for g in fsdp_groups(u) for p in g.fsdp_params) for u in self.units]
        from .phase_cache import ShardBindings
        self.state_bindings=ShardBindings(root)
        self.ledger=ForwardLedger();names={id(m):n for n,m in root.named_modules()}
        for unit in self.units:self.ledger.attach(names[id(unit)],unit,fsdp_groups(unit))
        sync(device);assert_sharded(root,config.cpu_offload if device.type=="cuda" else None)
        self.loaded_memory=memory(device)

    def call(self,command,args,kwargs):
        entry=time.perf_counter()
        sync(self.device)
        entry_sync_s=time.perf_counter()-entry
        start=time.perf_counter()
        self.ledger.reset()
        if self.device.type=="cuda":torch.cuda.reset_peak_memory_stats(self.device)
        from .memory_policy import plan_forward, available_memory
        # Native image/video expansion зависит от processor; неизвестная добавка
        # не выдумывается. Фактические expanded shapes публикуются после encoding.
        text_tokens=sum(len(row) for row in args[0].get("qwen3vl_32b",[]))
        hidden=self.root.network.model.embed_tokens.weight.shape[-1]
        snapshot=available_memory(self.device)
        plan=plan_forward(snapshot["free"],int(self.config.reserve_gib*2**30),text_tokens*hidden*24,
            text_tokens*hidden*8 if self.config.backend=="fsdp2_sequence" else 0,
            max(self.group_bytes,default=0),self.config.prefetch_blocks,self.config.memory_policy,
            reusable_bytes=snapshot["reusable_cache_bytes"])
        plan["allocator"]=snapshot["allocator"]
        plan["vision_expansion_estimate"]="UNKNOWN before native processor; estimate is a lower bound"
        if self.config.memory_policy=="auto":
            vote=torch.tensor([plan["effective_prefetch"],plan["mlp_budget_bytes"]],device=self.device,dtype=torch.int64)
            dist.all_reduce(vote,op=dist.ReduceOp.MIN)
            local_prefetch=plan["effective_prefetch"]
            plan["effective_prefetch"],plan["mlp_budget_bytes"]=vote.cpu().tolist()
            if plan["effective_prefetch"] < local_prefetch:
                plan["prefetch_reason"]="auto: a different rank had less estimated headroom (MIN consensus)"
            plan["prefetch_reduced"]=plan["effective_prefetch"] < plan["requested_prefetch"]
            for chain in (self.root.network.model.layers,self.root.network.visual.blocks):
                for i,unit in enumerate(chain):unit.set_modules_to_forward_prefetch(list(chain[i+1:i+1+plan["effective_prefetch"]]))
        signature=(plan["requested_prefetch"],plan["effective_prefetch"],plan["policy"],plan["prefetch_reason"])
        if getattr(self,"_last_prefetch_plan",None)!=signature:
            if dist.get_rank()==0:
                import json
                import sys
                print(json.dumps(dict(event="powershard_prefetch",command=command,
                    requested=signature[0],effective=signature[1],policy=signature[2],reason=signature[3])),file=sys.stderr)
            self._last_prefetch_plan=signature
        plan_s=time.perf_counter()-start
        self.memory_context=dict(plan)
        if getattr(self,"progress",None):
            self.progress("memory_plan",plan)
            self.memory_context["_progress"]=self.progress
        for layer in self.root.network.model.layers:
            layer.mlp._ps_memory_context=self.memory_context
            state=getattr(layer.self_attn,"_ps_qwen_sequence",{})
            state.update(kv_all_gathers=0,hidden_all_gathers=0,enabled=False)
        with torch.inference_mode(False),torch.no_grad():
            self.tracker.begin(self.device)
            compute_start=time.perf_counter()
            try:
                from .telemetry import region
                with region("Qwen/"+command):result=self.root(command,args,kwargs)
            finally:
                for unit in self.units:unit.reshard()
        compute_enqueue_s=time.perf_counter()-compute_start
        assert_sharded(self.root,self.config.cpu_offload if self.device.type=="cuda" else None)
        wait_start=time.perf_counter()
        sync(self.device)
        finish_wait_s=time.perf_counter()-wait_start
        finite_start=time.perf_counter()
        self.tracker.finish(result[0])
        finite_s=time.perf_counter()-finite_start
        from .topology import process_memory
        seq=[getattr(l.self_attn,"_ps_qwen_sequence",{}) for l in self.root.network.model.layers]
        active=any(s.get("enabled",False) for s in seq)
        metrics=dict(role="qwen",forward_s=time.perf_counter()-start,memory=memory(self.device),
            wall_phases=dict(entry_sync_s=entry_sync_s,memory_plan_s=plan_s,
                compute_enqueue_and_reshard_s=compute_enqueue_s,finish_cuda_wait_s=finish_wait_s,
                finite_validation_s=finite_s,
                note="Enqueue includes CPU/collective waits; phases are not exclusive GPU execution times."),
            cpu_memory=process_memory(),world_size=self.world,inter_gpu_sharding=self.world>1,
            sharded_after_forward=True,duplicated_compute=self.world>1 and not active,
            backend="fsdp2_sequence",vision_compute="replicated native vision, sharded weights",
            text_sequence="local attention/MLP; global KV + hidden gather per layer for native DeepStack" if active else "replicated",
            checkpoint=self.identity,attention=self.attention.report(),execution=self.ledger.report(),
            persistent_shard_bytes=self.shard_bytes,cpu_shard_bytes=self.shard_bytes if self.config.cpu_offload else 0,
            weight_placement=self.config.weight_placement,managed_shard_bytes=self.shard_bytes if self.managed_pool else 0,
            ats=self.managed_pool.report() if self.managed_pool else None,
            condition_shape=list(result[0].shape),condition_dtype=str(result[0].dtype),
            memory_plan=plan,mlp={str(i):l.mlp._ps_mlp_report for i,l in enumerate(self.root.network.model.layers) if hasattr(l.mlp,"_ps_mlp_report")},
            sequence_communication=dict(kv_all_gathers=sum(s.get("kv_all_gathers",0) for s in seq),hidden_all_gathers=sum(s.get("hidden_all_gathers",0) for s in seq)),
            text_attention_shapes={str(i):l.self_attn._ps_shapes for i,l in enumerate(self.root.network.model.layers) if hasattr(l.self_attn,"_ps_shapes")})
        return result,metrics

    def idle(self):
        """CPU shards сохраняются; CUDA allocator освобождается один раз между фазами."""
        if not self.config.cpu_offload:
            raise ValueError("Сохранить CPU shards можно только с CPUOffloadPolicy")
        before=memory(self.device)
        for unit in self.units:unit.reshard()
        assert_sharded(self.root,True)
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device);torch.cuda.empty_cache()
        return dict(before=before,after=memory(self.device),cpu_shard_bytes=self.shard_bytes,
                    retained="CPU shards + small CUDA buffers/NCCL context")
