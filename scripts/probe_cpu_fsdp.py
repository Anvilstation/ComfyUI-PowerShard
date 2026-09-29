#!/usr/bin/env python3
"""Отдельная CPU/Gloo проверка настоящего FSDP2, НЕ CUDA/NCCL приёмка."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument("--worker",type=int);p.add_argument("--store");p.add_argument("--world-size",type=int,default=3);p.add_argument("--output",default="reports/cpu-fsdp.json");a=p.parse_args()
if a.world_size<1:p.error('world-size должен быть положительным')
if a.worker is None:
 with tempfile.TemporaryDirectory() as d:
  procs=[]
  for i in range(a.world_size):
   procs.append(subprocess.Popen([sys.executable,__file__,"--worker",str(i),"--world-size",str(a.world_size),"--store",d+"/store","--output",d+f"/rank{i}.json"]))
  try:
   for child in procs:
    if child.wait(timeout=120):raise RuntimeError("CPU FSDP worker failed")
   result={"device":"CPU x86_64","backend":"Gloo","CUDA_NCCL":"NOT_RUN","ranks":[json.load(open(d+f"/rank{i}.json")) for i in range(a.world_size)]}
   Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print("CPU/Gloo FSDP2 PASS:",a.output)
  finally:
   for c in procs:
    if c.poll() is None:c.kill();c.wait()
else:
 import torch
 import torch.distributed as dist
 from datetime import timedelta
 from powershard.validation import distributed_probe
 torch.set_num_threads(1)
 dist.init_process_group("gloo",init_method=Path(a.store).resolve().as_uri(),rank=a.worker,world_size=a.world_size,timeout=timedelta(seconds=60))
 try:Path(a.output).write_text(json.dumps(distributed_probe(torch.device("cpu")),indent=2))
 finally:dist.destroy_process_group()
