#!/usr/bin/env python3
"""Pageable/pinned H2D+D2H, отдельный безопасный subprocess для каждой из трёх GPU."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2');p.add_argument('--numa-policy',choices=['none','auto','bind'],default='none')
p.add_argument('--mib',type=int,default=64);p.add_argument('--iterations',type=int,default=10);p.add_argument('--timeout',type=int,default=120)
p.add_argument('--output',default='reports/local-transfers.json');p.add_argument('--worker',type=int);p.add_argument('--uuid');a=p.parse_args()
if a.mib<1 or a.iterations<1:p.error('mib и iterations должны быть положительными')
from powershard.topology import numa_launch_prefix,configure_worker_affinity,transfer_benchmark,process_memory
if a.worker is None:
 from powershard.runtime import resolve_gpus
 uuids=resolve_gpus(tuple(a.gpus.split(',')));env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(uuids)
 rows=[]
 with tempfile.TemporaryDirectory(prefix='powershard-bw-') as folder:
  # Последовательно: измеряем каждый link без конкурирующих transfers.
  for rank,uuid in enumerate(uuids):
   output=Path(folder)/f'rank{rank}.json'
   cmd=numa_launch_prefix(uuid,a.numa_policy)+[sys.executable,__file__,'--worker',str(rank),'--uuid',uuid,
        '--numa-policy',a.numa_policy,'--mib',str(a.mib),'--iterations',str(a.iterations),'--output',str(output)]
   subprocess.run(cmd,env=env,check=True,timeout=a.timeout)
   rows.append(json.loads(output.read_text()))
 report={'status':'PASS','ranks':rows,'note_ru':'Измеренный transfer, НЕ доказательство конкретного маршрута NVLink; сопоставить diagnose/topology. Нет NCCL/FSDP в этом microbenchmark.'}
else:
 locality=configure_worker_affinity(a.uuid,a.numa_policy)
 import torch
 torch.cuda.set_device(a.worker);device=torch.device('cuda',a.worker)
 report={'rank':a.worker,'uuid':a.uuid,'gpu':torch.cuda.get_device_name(device),'locality':locality,
         'measurements':transfer_benchmark(device,a.mib,a.iterations),'cpu_memory':process_memory()}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2));print(a.output)
