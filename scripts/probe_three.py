#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from powershard.config import DistributedConfig
from powershard.runtime import Session
p=argparse.ArgumentParser();p.add_argument("--profile");p.add_argument("--gpus");p.add_argument("--timeout",type=int);p.add_argument("--reports",default="reports/local-probe")
p.add_argument("--cpu-offload",action=argparse.BooleanOptionalAction,default=None)
p.add_argument("--pin-memory",action=argparse.BooleanOptionalAction,default=None)
p.add_argument("--prefetch-blocks",type=int,choices=[0,1,2]);p.add_argument("--numa-policy",choices=["none","auto","bind"])
p.add_argument('--attention-backend',choices=['auto','sdpa','flash_attn','vllm_flash_attn','sageattention','math']);p.add_argument('--allow-fallback',action=argparse.BooleanOptionalAction,default=None)
p.add_argument('--memory-profile',choices=['custom','ram_min'],default=None)
p.add_argument('--workspace-mib',type=int,default=None)
p.add_argument('--stage-cache-mib',type=int,default=None)
a=p.parse_args()
options=json.loads(Path(a.profile).read_text()) if a.profile else {"timeout_s":120}
if a.gpus is not None:options["gpu_ids"]=tuple(a.gpus.split(","))
if a.timeout is not None:options["timeout_s"]=a.timeout
for key in ("cpu_offload","pin_memory","prefetch_blocks","numa_policy","attention_backend","allow_fallback","memory_profile","workspace_mib","stage_cache_mib"):
 if getattr(a,key) is not None:options[key]=getattr(a,key)
options["allow_unverified"]=True
c=DistributedConfig(**options)
s=Session(None,c,Path.cwd(),a.reports,probe_only=True)
try:
 s.start(); print(json.dumps(s.history,indent=2))
finally:s.close()
