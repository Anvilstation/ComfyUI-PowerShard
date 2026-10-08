"""Optional standalone JIT smoke. Not imported by workers or production nodes."""
import triton
import triton.language as tl
import torch


@triton.jit
def _add(x,y,z,n:tl.constexpr,BLOCK:tl.constexpr):
    indexes=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    left=tl.load(x+indexes,mask=indexes<n,other=0.)
    right=tl.load(y+indexes,mask=indexes<n,other=0.)
    tl.store(z+indexes,left+right,mask=indexes<n)


def add(x,y):
    out=torch.empty_like(x)
    _add[(triton.cdiv(x.numel(),256),)](x,y,out,x.numel(),BLOCK=256)
    return out
