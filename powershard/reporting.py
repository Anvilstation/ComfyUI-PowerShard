"""JSON manifests без prompt/tensor dump по умолчанию."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def describe_options(value):
    """Отчёт чужого sampler не должен сериализовать callbacks/tensors или ломать fallback."""
    if value is None or type(value) in (str,bool,int,float):return value
    if isinstance(value,dict):return {str(k):describe_options(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [describe_options(v) for v in value]
    if hasattr(value,"shape") and hasattr(value,"dtype"):
        return dict(shape=list(value.shape),dtype=str(value.dtype),contents="NOT_RECORDED")
    return dict(type=type(value).__module__+"."+type(value).__qualname__,serialized=False)


def redact(value,keep_inputs=False):
    if isinstance(value,dict):
        return {k:({"sha256":hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                     "redacted":True} if not keep_inputs and (k.lower() in ("text","prompt","positive","negative") or "prompt_text" in k.lower())
                   else redact(v,keep_inputs)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [redact(v,keep_inputs) for v in value]
    return value


def environment_identity():
    from . import __version__
    try:
        commit=subprocess.check_output(["git","-C",str(Path(__file__).resolve().parents[1]),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL,timeout=3).strip()
    except (OSError,subprocess.SubprocessError):commit="UNKNOWN"
    torch=sys.modules.get("torch")
    return dict(powershard=__version__,commit=commit,python=sys.version,architecture=platform.machine(),
                torch=getattr(torch,"__version__",None),torch_cuda=getattr(getattr(torch,"version",None),"cuda",None))
