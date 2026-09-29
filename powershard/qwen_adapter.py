"""Настоящий comfy.sd.CLIP с native tokenizer и worker encoder, без weights в host."""
import copy
from dataclasses import replace
from pathlib import Path
import warnings
import torch
import comfy.sd
import comfy.model_patcher
import comfy.model_management
from .runtime import Session
from .qwen import QwenConfig,infer_qwen_config,qwen_memory_plan
from .checkpoint import Checkpoint,memory_plan
from .conditioning_cache import ConditioningCache,content_hash


class QwenProxy(torch.nn.Module):
    def __init__(self,session,options,tokenizer_identity):
        super().__init__()
        self.session,self.options,self.tokenizer_identity=session,options,tokenizer_identity
        self.clip_options={};self.dtypes={torch.float16};self.device=torch.device("cpu")
        self.cache=ConditioningCache(options.cache_mib*2**20)
        self.last_encoding={}

    def reset_clip_options(self):self.clip_options={}
    def set_clip_options(self,options):self.clip_options.update(options)
    def memory_estimation_function(self,tokens,device=None):return 0 # worker memory tracked by patcher, not host weights

    def encode_token_weights(self,tokens):
        import time
        from .web_api import provider_stamp
        start=time.perf_counter()
        options={k:v for k,v in self.clip_options.items() if k!="execution_device"}
        path=Path(self.session.checkpoint);stat=path.stat()
        key=content_hash(dict(tokens=tokens,options=options,checkpoint=(str(path),stat.st_size,stat.st_mtime_ns),
            tokenizer=self.tokenizer_identity,config=self.session.config.to_dict(),provider=provider_stamp(),role="qwen",
            role_options=self.session.role_options))
        result=self.cache.get(key)
        hit=result is not None
        if not hit:
            result=self.session.call("encode",(tokens,),dict(clip_options=options),
                cancel=comfy.model_management.throw_exception_if_processing_interrupted)
            self.cache.put(key,result)
        if self.options.idle_policy=="release":self.session.close()
        elif self.options.idle_policy=="cpu_shards":self.session.idle()
        self.last_encoding=dict(cache_hit=hit,encoding_wall_s=time.perf_counter()-start,cache=self.cache.report(),
                                conditioning_hash=key,idle_policy=self.options.idle_policy)
        return result


class QwenPatcher(comfy.model_patcher.ModelPatcher):
    @property
    def session(self):return self.model.session

    def validate(self):
        from .wire import has_effect
        if self.patches or self.hook_patches or self.weight_wrapper_patches or self.injections or self.forced_hooks or self.additional_models or self.object_patches:
            raise ValueError("Qwen distributed encoder: LoRA/weight/object/hooks patches не сериализованы; encoding не запущен")
        if any(v for family in (self.wrappers,self.callbacks) for groups in family.values() for v in groups.values()):
            raise ValueError("Qwen distributed encoder: сторонний callback/wrapper не сериализуется")
        if has_effect(self.model_options):
            raise ValueError("Qwen distributed encoder: model_options patch не сериализован; encoding не запущен")

    def add_patches(self,patches,*args,**kwargs):
        if patches:raise ValueError("Qwen LoRA adapter пока не реализован; patches не игнорируются")
        return []

    def clone(self,*args,**kwargs):
        other=super().clone(*args,**kwargs)
        other.model=QwenProxy(self.session,self.model.options,self.model.tokenizer_identity)
        other.model.clip_options=copy.deepcopy(self.model.clip_options)
        if hasattr(self,"powershard_memory_plan"):
            other.powershard_memory_plan=copy.deepcopy(self.powershard_memory_plan)
        return other

    def load(self,device_to=None,**kwargs):
        self.validate();self.model.device=device_to or self.load_device
        self.model.current_patcher=self;self.model.model_loaded_weight_memory=self.model_size()

    def loaded_size(self):
        if not self.session.running:
            return 0
        # Аналогично H3: до первого RPC — бюджет из qwen memory plan, не ноль,
        # иначе ComfyUI-менеджер перестаёт резервировать VRAM и забивает пул.
        plan = getattr(self, "powershard_memory_plan", None)
        return self.session.last_memory or (plan or {}).get("host_gpu_parameter_budget_bytes", 0)

    def partially_load(self,device_to,extra_memory=0,force_patch_weights=False):
        self.patch_model(device_to=device_to);return 0

    def partially_unload(self,device_to,memory_to_free=0,force_patch_weights=False):
        size=self.loaded_size();self.session.deactivate();return size

    def unpatch_model(self,device_to=None,unpatch_weights=True):
        self.session.deactivate()
        return super().unpatch_model(device_to=device_to,unpatch_weights=unpatch_weights)


class DistributedQwenCLIP(comfy.sd.CLIP):
    def clone(self,disable_dynamic=False):
        other=DistributedQwenCLIP(no_init=True)
        other.patcher=self.patcher.clone(disable_dynamic=disable_dynamic)
        other.cond_stage_model=other.patcher.model
        other.tokenizer=self.tokenizer
        other.layer_idx=self.layer_idx;other.tokenizer_options=self.tokenizer_options.copy()
        other.use_clip_schedule=self.use_clip_schedule;other.apply_hooks_to_conds=self.apply_hooks_to_conds
        return other

    def load_model(self,tokens={}):
        self.patcher.validate()
        return super().load_model(tokens)

    def generate(self,*args,**kwargs):
        raise ValueError("H3 Qwen — truncated conditioning encoder без lm_head; generate() не определён")

    def clear_cache(self):self.cond_stage_model.cache.clear()

    def state_dict_for_saving(self):
        raise ValueError("Remote CLIP не содержит весов в host; исходный checkpoint сохранён отдельно")

    def get_sd(self):
        return self.state_dict_for_saving()


def load_qwen(path,config,options=None,embedding_directory=None,report_dir=None):
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    import comfy.text_encoders.minimax as native
    from .devices import resolve_gpu_selection
    options=options or QwenConfig()
    ckpt=Checkpoint(path);infer_qwen_config(ckpt.tensors)
    quant=ckpt.quantization()
    if bool(quant)!=(config.precision=="int8_fp16"):
        raise ValueError("Выберите Qwen precision, соответствующий storage checkpoint")
    if options.idle_policy=="cpu_shards" and not config.cpu_offload:
        warnings.warn("Qwen idle=cpu_shards включает CPUOffloadPolicy также во время encoding; эффективный cpu_offload=True")
        config=replace(config,cpu_offload=True)
    selected=resolve_gpu_selection(config.gpu_ids)
    comfy_path=Path(comfy.sd.__file__).resolve().parents[1]
    session=Session(str(ckpt.path),config,comfy_path,report_dir,role="qwen",role_options=options.to_dict())
    tokenizer=MiniMaxH3Tokenizer(embedding_directory=embedding_directory)
    token_dir=Path(native.__file__).parent/"qwen25_tokenizer"
    import hashlib
    tokenizer_identity=content_hash([(p.name,hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(token_dir.iterdir()) if p.is_file()])
    proxy=QwenProxy(session,options,tokenizer_identity)
    plan=qwen_memory_plan(ckpt.tensors,len(selected),config.prefetch_blocks,config.cpu_offload)
    size=plan["host_gpu_parameter_budget_bytes"]
    clip=DistributedQwenCLIP(no_init=True)
    clip.patcher=QwenPatcher(proxy,torch.device("cuda",int(selected[0]["user_id"])),torch.device("cpu"),size=size)
    clip.patcher.powershard_memory_plan=plan
    clip.cond_stage_model=proxy;clip.tokenizer=tokenizer
    clip.layer_idx=None;clip.use_clip_schedule=False;clip.apply_hooks_to_conds=None;clip.tokenizer_options={}
    return clip
