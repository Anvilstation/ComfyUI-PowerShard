#!/usr/bin/env python3
"""Настоящая малая native H3 + FP16 Safe + FSDP2, NCCL, 1 diagnostic / 3 acceptance ranks.

Случайные малые веса: проверка execution/FSDP path, НЕ генерация реального H3.
"""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--comfy',required=True);p.add_argument('--gpus',default='0,1,2')
p.add_argument('--single-rank',action='store_true');p.add_argument('--cpu-offload',action='store_true')
p.add_argument('--pin-memory',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--prefetch-blocks',type=int,choices=[0,1,2],default=0)
p.add_argument('--numa-policy',choices=['none','auto','bind'],default='none');p.add_argument('--timeout',type=int,default=180)
p.add_argument('--output',default='reports/local-h3-cuda-probe.json');p.add_argument('--worker',type=int);p.add_argument('--folder');p.add_argument('--uuid');a=p.parse_args()
world=int(os.environ.get('POWERSHARD_PROBE_WORLD','1'))
if a.worker is None:
 from powershard.runtime import resolve_gpus
 from powershard.topology import numa_launch_prefix
 uuids=resolve_gpus(tuple(a.gpus.split(',')));env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(uuids)
 if a.single_rank and len(uuids)!=1:p.error('--single-rank требует явно выбрать одну GPU; набор не сокращается автоматически')
 world=len(uuids);env['POWERSHARD_PROBE_WORLD']=str(world)
 env['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'
 with tempfile.TemporaryDirectory(prefix='powershard-h3-probe-') as folder:
  procs=[];logs=[]
  try:
   for rank in range(world):
    log=Path(folder,f'rank{rank}.log').open('w');logs.append(log)
    child=numa_launch_prefix(uuids[rank],a.numa_policy)+[sys.executable,__file__,*sys.argv[1:],
            '--worker',str(rank),'--folder',folder,'--uuid',uuids[rank]]
    procs.append(subprocess.Popen(child,env=env,stdout=log,stderr=log))
   import time
   deadline=time.monotonic()+a.timeout
   pending=set(range(world))
   while pending:
    if time.monotonic()>deadline:raise TimeoutError('H3 FSDP smoke timeout')
    for rank in list(pending):
     code=procs[rank].poll()
     if code is not None:
      if code:raise RuntimeError(f'rank {rank} failed')
      pending.remove(rank)
    time.sleep(.05)
   report={'status':'PASS','world_size':world,'real_checkpoint':'NOT_RUN','ranks':[json.loads(Path(folder,f'rank{rank}.json').read_text()) for rank in range(world)]}
  except BaseException as e:
   report={'status':'FAIL','error':str(e),'world_size':world}
   raise
  finally:
   for proc in procs:
    if proc.poll() is None:proc.terminate()
   for proc in procs:
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
   for log in logs:log.close()
   report['logs']={str(rank):Path(folder,f'rank{rank}.log').read_text() for rank in range(len(logs))}
   Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2))
else:
 from powershard.topology import configure_worker_affinity,process_memory
 locality=configure_worker_affinity(a.uuid,a.numa_policy)
 sys.path.insert(0,str(Path(a.comfy).resolve()));sys.argv=['h3-probe','--disable-dynamic-vram','--use-pytorch-cross-attention']
 import comfy.options
 comfy.options.enable_args_parsing()
 import torch,torch.distributed as dist
 from datetime import timedelta
 from powershard.preflight import collectives
 from powershard.validation import distributed_probe,native_h3_probe
 from powershard.config import DistributedConfig
 from powershard.source_guard import verify_comfy
 torch.cuda.set_device(a.worker);device=torch.device('cuda',a.worker)
 dist.init_process_group('nccl',init_method=Path(a.folder,'store').as_uri(),rank=a.worker,world_size=world,
                         timeout=timedelta(seconds=a.timeout),device_id=device)
 try:
  caps=verify_comfy(a.comfy)
  config=DistributedConfig(cpu_offload=a.cpu_offload,pin_memory=a.pin_memory,prefetch_blocks=a.prefetch_blocks,reserve_gib=0)
  report={'rank':a.worker,'locality':locality,'capabilities':caps,'torch':torch.__version__,'cuda':torch.version.cuda}
  report['collectives']=collectives(device)
  report['tiny_linear_fsdp']=distributed_probe(device,a.cpu_offload,a.pin_memory,a.prefetch_blocks,expected_world=world)
  report['native_h3_fsdp']=native_h3_probe(device,config,a.folder)
  report['cpu_memory']=process_memory()
  Path(a.folder,f'rank{a.worker}.json').write_text(json.dumps(report,indent=2))
 finally:dist.destroy_process_group()
