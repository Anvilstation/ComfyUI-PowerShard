"""Внутренний dense/packed adapter пользовательского vllm_flash_attn.

lengths — CPU tuple метаданных вызывающей стороны: без чтения cu_seqlens с GPU
на каждом блоке. Они задают точные границы, без padded tokens.
"""
import torch
from .attention_contract import AttentionOptions
from .attention_providers import FlashProvider


class VLLMFlashAdapter(FlashProvider):
    def __init__(self,module=None):
        super().__init__("vllm_flash_attn",module)

    def packed(self,q,k,v,*,lengths_q,lengths_k,options=None):
        o=options or AttentionOptions()
        if not isinstance(lengths_q,(tuple,list)) or not isinstance(lengths_k,(tuple,list)):
            raise ValueError("lengths_q/k должны быть CPU list/tuple целых длин, не CUDA tensor")
        if len(lengths_q)!=len(lengths_k) or not lengths_q or any(type(n) is not int or n<=0 for n in (*lengths_q,*lengths_k)):
            raise ValueError("Каждая varlen sequence должна иметь положительную длину Q/K")
        if q.ndim!=3 or k.ndim!=3 or v.ndim!=3 or sum(lengths_q)!=q.shape[0] or sum(lengths_k)!=k.shape[0] or k.shape[:2]!=v.shape[:2]:
            raise ValueError("Packed BLHD boundaries не соответствуют tensors")
        a=c=0
        for lq,lk in zip(lengths_q,lengths_k):
            self.check(q[a:a+lq].unsqueeze(0),k[c:c+lk].unsqueeze(0),v[c:c+lk].unsqueeze(0),o)
            a+=lq;c+=lk
        if self.varlen is None:
            from .attention_contract import UnsupportedAttention
            raise UnsupportedAttention("Установленная сборка не предоставляет varlen entrypoint")
        cuq=torch.tensor((0,*lengths_q),device=q.device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
        cuk=torch.tensor((0,*lengths_k),device=q.device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
        out=self.varlen(q.contiguous(),k.contiguous(),v.contiguous(),cu_seqlens_q=cuq,cu_seqlens_k=cuk,
                       max_seqlen_q=max(lengths_q),max_seqlen_k=max(lengths_k),**self.kwargs(o,"varlen"))
        out=out[0] if isinstance(out,(tuple,list)) else out
        if not isinstance(out,torch.Tensor) or out.shape!=(q.shape[0],q.shape[1],v.shape[-1]) or out.dtype!=q.dtype or out.device!=q.device:
            raise RuntimeError("vllm_flash_attn нарушил packed output contract")
        return out
