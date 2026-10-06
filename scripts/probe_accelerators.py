#!/usr/bin/env python3
"""Each provider runs in its own context; no package installs or stack changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus",default="0",help="CUDA-visible IDs/UUIDs; each tested in isolation")
    parser.add_argument("--providers",default="flash_attn,vllm_flash_attn,sdpa,triton")
    parser.add_argument("--mode",choices=["token","ulysses"],default="token")
    parser.add_argument("--world",type=int,default=5)
    parser.add_argument("--total",type=int,default=46535)
    parser.add_argument("--heads",type=int,default=56)
    parser.add_argument("--dim",type=int,default=128)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--output",default="accelerators.json")
    args=parser.parse_args()
    if min(args.world,args.total,args.heads,args.dim,args.repeats)<1:parser.error("sizes must be positive")
    providers=args.providers.split(",")
    if any(p not in ("flash_attn","vllm_flash_attn","sdpa","triton") for p in providers):parser.error("unknown provider")
    from powershard.devices import resolve_gpu_selection
    devices=resolve_gpu_selection(tuple(args.gpus.split(",")))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    reports=[]
    for device in devices:
        for provider in providers:
            print(f"Testing {device['user_id']} {device['uuid']} {provider} {args.mode}",file=sys.stderr,flush=True)
            env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=device["uuid"]
            env["PYTHONPATH"]=os.pathsep.join([str(Path(__file__).resolve().parents[1]),env.get("PYTHONPATH","")])
            cmd=[sys.executable,"-m","powershard.accelerator_probe",provider,"--world",str(args.world),
                 "--mode",args.mode,"--total",str(args.total),"--heads",str(args.heads),"--dim",str(args.dim),"--repeats",str(args.repeats)]
            try:
                run=subprocess.run(cmd,env=env,capture_output=True,text=True)
            except KeyboardInterrupt:
                output.write_text(json.dumps(dict(status="INTERRUPTED",results=reports),indent=2));raise
            try:report=json.loads(run.stdout)
            except ValueError:report=dict(status="FAIL",reason="subprocess did not return JSON",stdout=run.stdout[-2000:])
            report.update(gpu=device,exit_code=run.returncode,stderr=run.stderr[-5000:])
            reports.append(report)
            output.write_text(json.dumps(dict(status="COMPLETE",results=reports),indent=2))
            print(f"{provider}: {report['status']}; median {report.get('median_cuda_ms','—')} ms",file=sys.stderr,flush=True)
    return 0 if all(r["status"]=="PASS" for r in reports) else 2


if __name__=="__main__":raise SystemExit(main())
