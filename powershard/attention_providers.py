"""Lazy instance adapters. Никакого shadowing установленного flash_attn."""
import importlib
import importlib.metadata
import inspect
import warnings
from pathlib import Path
import torch
from .attention_contract import UnsupportedAttention, validate


def module_identity(name, module=None):
    module = importlib.import_module(name) if module is None else module
    distributions = importlib.metadata.packages_distributions().get(name, [])
    versions = {d:importlib.metadata.version(d) for d in distributions}
    path = str(getattr(module,"__file__", "unknown"))
    stat = Path(path).stat() if Path(path).is_file() else None
    return dict(module=name,path=path,distributions=versions,
                declared_version=getattr(module,"__version__",None),
                file_identity=[stat.st_size,stat.st_mtime_ns] if stat else None,
                legacy_shim="flash_attn_shim" in path or "shim" in str(getattr(module,"__version__","")))


class FlashProvider:
    def __init__(self, name="flash_attn", module=None):
        self.name = name
        self.module = importlib.import_module(name) if module is None else module
        self.identity = module_identity(name,self.module)
        if self.identity["legacy_shim"]:
            warnings.warn("Признаки глобального flash_attn_shim в path/version: проверьте origin. Допуск определяется numerical probe, не строкой версии.")
        self.varlen = getattr(self.module,"flash_attn_varlen_func",None)
        self.dense = getattr(self.module,"flash_attn_func",None)
        if not callable(self.varlen) and not callable(self.dense):
            raise UnsupportedAttention(f"{name} не экспортирует dense/varlen attention")
        # Единожды. **kwargs не считаем доказательством поддержки конкретного поля.
        self.signatures = {}
        for mode,fn in (("dense",self.dense),("varlen",self.varlen)):
            if fn is not None:
                try:self.signatures[mode] = inspect.signature(fn)
                except (TypeError,ValueError) as error:raise UnsupportedAttention(f"Нельзя проверить {name}.{mode} signature: {error}") from error
        self.identity["signatures"] = {k:str(v) for k,v in self.signatures.items()}

    def kwargs(self,o,mode):
        parameters = self.signatures[mode].parameters
        values = dict(softmax_scale=o.softmax_scale,dropout_p=o.dropout_p,causal=o.causal,
                      window_size=o.window_size,alibi_slopes=o.alibi_slopes,
                      deterministic=o.deterministic,return_attn_probs=o.return_attn_probs)
        defaults = dict(softmax_scale=None,dropout_p=0.,causal=False,window_size=(-1,-1),
                        alibi_slopes=None,deterministic=False,return_attn_probs=False)
        out = {}
        for key,value in values.items():
            if key in parameters and parameters[key].kind != inspect.Parameter.POSITIONAL_ONLY:
                out[key] = value
            elif (value is not None if key == "alibi_slopes" else value != defaults[key]):
                raise UnsupportedAttention(f"{self.name}.{mode}: signature не подтверждает {key}")
        return out

    def check(self,q,k,v,o):
        validate(q,k,v,o)
        if o.mask is not None:raise UnsupportedAttention("Flash API не принимает произвольный mask")
        if o.return_attn_probs:raise UnsupportedAttention("Flash S_dmask не равен нормализованным probabilities; требуется math")
        if q.shape[-1] != v.shape[-1]:raise UnsupportedAttention("Этот Flash adapter требует Dv=Dq; требуется fallback")
        if q.dtype not in (torch.float16,torch.bfloat16):raise UnsupportedAttention("Flash provider требует FP16/BF16 inputs")
        positional = o.causal or o.window_size != (-1,-1) or o.alibi_slopes is not None
        if positional and q.shape[1] != k.shape[1] and o.causal_alignment != "bottom_right":
            raise UnsupportedAttention("Flash rectangular positional masks выровнены bottom-right, запрос upper-left")

    def __call__(self,q,k,v,o):
        self.check(q,k,v,o)
        # vLLM custom build: предпочитается его реальный varlen entrypoint.
        use_varlen = self.varlen is not None and (self.name == "vllm_flash_attn" or self.dense is None)
        if use_varlen:
            b,lq,hq,dq = q.shape
            _,lk,hk,dk = k.shape
            dv = v.shape[-1]
            cuq = torch.arange(b+1,device=q.device,dtype=torch.int32)*lq
            cuk = torch.arange(b+1,device=q.device,dtype=torch.int32)*lk
            args = self.kwargs(o,"varlen")
            for key in ("cu_seqlens_q","cu_seqlens_k","max_seqlen_q","max_seqlen_k"):
                if key not in self.signatures["varlen"].parameters:raise UnsupportedAttention(f"Нет обязательного varlen аргумента {key}")
            out = self.varlen(q.reshape(b*lq,hq,dq).contiguous(),k.reshape(b*lk,hk,dk).contiguous(),
                v.reshape(b*lk,hk,dv).contiguous(),cu_seqlens_q=cuq,cu_seqlens_k=cuk,
                max_seqlen_q=lq,max_seqlen_k=lk,**args)
            expected = (b*lq,hq,dv)
        else:
            out = self.dense(q.contiguous(),k.contiguous(),v.contiguous(),**self.kwargs(o,"dense"))
            expected = (q.shape[0],q.shape[1],q.shape[2],v.shape[-1])
        # Некоторые сборки всегда возвращают (out,lse,...), даже без return flag.
        out = out[0] if isinstance(out,(tuple,list)) else out
        if not isinstance(out,torch.Tensor) or tuple(out.shape)!=expected or out.dtype!=q.dtype or out.device!=q.device:
            raise RuntimeError(f"{self.name}: нарушен output tensor contract")
        return out.reshape(q.shape[0],q.shape[1],q.shape[2],v.shape[-1])


class SageProvider:
    def __init__(self):
        module = importlib.import_module("sageattention")
        self.fn = module.sageattn
        self.identity = module_identity("sageattention",module)
        self.parameters = inspect.signature(self.fn).parameters
        self.identity["signature"] = str(inspect.signature(self.fn))

    def __call__(self,q,k,v,o):
        validate(q,k,v,o)
        if o.mask is not None or o.dropout_p or o.alibi_slopes is not None or o.window_size != (-1,-1) or o.return_attn_probs or o.deterministic:
            raise UnsupportedAttention("Sage adapter: mask/dropout/window/ALiBi/return/deterministic требуют другого пути")
        if q.shape[2]!=k.shape[2] or q.shape[-1]!=v.shape[-1]:raise UnsupportedAttention("Sage adapter: GQA/Dv требует другого пути")
        if o.causal and q.shape[1]!=k.shape[1]:raise UnsupportedAttention("Sage rectangular causal не подтверждён")
        args = dict(tensor_layout="NHD",is_causal=o.causal,sm_scale=o.softmax_scale)
        if not set(args)<=set(self.parameters):raise UnsupportedAttention("Sage signature не подтверждает layout/causal/scale")
        out = self.fn(q.contiguous(),k.contiguous(),v.contiguous(),**args)
        if out.shape!=q.shape or out.dtype!=q.dtype:raise RuntimeError("Sage output contract нарушен")
        return out
