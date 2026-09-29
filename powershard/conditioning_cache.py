"""Ограниченный CPU LRU; keys хешируют содержимое inputs, а не адреса tensors."""
from collections import OrderedDict
import hashlib
import json
import torch
import weakref

_CACHES = weakref.WeakSet()


def content_hash(value):
    h=hashlib.sha256()
    def framed(tag, data):
        h.update(tag);h.update(len(data).to_bytes(8,"big"));h.update(data)
    def walk(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous()
            framed(b"T",str((t.dtype,tuple(t.shape))).encode())
            framed(b"D",t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            if not all(type(k) is str for k in x):raise TypeError("Cache dict keys must be strings")
            framed(b"M",str(len(x)).encode())
            for k in sorted(x):framed(b"K",k.encode());walk(x[k])
        elif isinstance(x,(list,tuple)):
            framed(b"L" if isinstance(x,list) else b"U",str(len(x)).encode())
            for a in x:walk(a)
        elif x is None or type(x) in (bool,int,float,str):framed(type(x).__name__.encode(),json.dumps(x,allow_nan=False).encode())
        else:raise TypeError(f"Не поддерживается cache input {type(x).__name__}")
    walk(value)
    return h.hexdigest()


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(v) for v in value)
    return 0


class ConditioningCache:
    def __init__(self,limit):
        self.limit=limit;self.entries=OrderedDict();self.bytes=0;self.hits=0;self.misses=0
        _CACHES.add(self)
    def get(self,key):
        if key not in self.entries:self.misses+=1;return None
        self.hits+=1;value=self.entries.pop(key);self.entries[key]=value
        return cpu_copy(value)
    def put(self,key,value):
        size=tensor_bytes(value)
        if not self.limit or size>self.limit:return
        if key in self.entries:self.bytes-=tensor_bytes(self.entries.pop(key))
        while self.entries and self.bytes+size>self.limit:
            _,old=self.entries.popitem(last=False);self.bytes-=tensor_bytes(old)
        self.entries[key]=cpu_copy(value);self.bytes+=size
    def clear(self):self.entries.clear();self.bytes=0
    def report(self):return dict(bytes=self.bytes,limit_bytes=self.limit,entries=len(self.entries),hits=self.hits,misses=self.misses)


def clear_all_caches():
    for cache in list(_CACHES):cache.clear()
