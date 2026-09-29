#!/usr/bin/env python3
"""Изолированный kernel+packing microbenchmark, НЕ H3 end-to-end benchmark."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('--gpus',default='all');p.add_argument('--backend',default='sdpa')
p.add_argument('--length',type=int,default=256);p.add_argument('--head-dim',type=int,default=128);p.add_argument('--heads',type=int,default=56)
p.add_argument('--iterations',type=int,default=10);p.add_argument('--output',default='reports/attention-benchmark.json')
p.add_argument('--worker',action='store_true');a=p.parse_args()
if min(a.length,a.head_dim,a.heads,a.iterations)<1:p.error('Размеры и iterations должны быть положительными')
if not a.worker:
 from powershard.devices import visible_inventory,resolve_gpu_selection
 from powershard.attention_policy import prepare_policy
 from powershard.config import DistributedConfig
 inventory=visible_inventory()
 report=dict(status='NOT_RUN',reason='Нет доступных CUDA GPU',ranks=[])
 if inventory:
  selected=resolve_gpu_selection(a.gpus,inventory)
  policy=prepare_policy(DistributedConfig(gpu_ids=a.gpus,attention_backend=a.backend,allow_fallback=False),selected,
                        geometry=dict(head_dim=a.head_dim,heads=a.heads))
  with tempfile.TemporaryDirectory(prefix='powershard-attention-bench-') as folder:
   for rank,d in enumerate(selected):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=d['uuid']
    output=Path(folder,f'{rank}.json')
    subprocess.run([sys.executable,__file__,'--worker','--backend',policy['effective'],'--length',str(a.length),
      '--head-dim',str(a.head_dim),'--heads',str(a.heads),'--iterations',str(a.iterations),'--output',str(output)],check=True,env=env,timeout=180)
    report['ranks'].append(dict(rank=rank,uuid=d['uuid'],**json.loads(output.read_text())))
  report.update(status='PASS',policy=policy);report.pop('reason',None)
else:
 import time,torch
 from powershard.config import DistributedConfig
 from powershard.attention_policy import AttentionDispatcher
 from powershard.topology import process_memory
 torch.cuda.set_device(0)
 g=torch.Generator(device='cuda').manual_seed(44)
 q,k,v=[torch.randn(1,a.length,a.heads,a.head_dim,device='cuda',dtype=torch.float16,generator=g) for _ in range(3)]
 dispatcher=AttentionDispatcher(DistributedConfig(attention_backend=a.backend,allow_fallback=False))
 for _ in range(3):dispatcher(q,k,v)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 t=time.perf_counter();start.record()
 for _ in range(a.iterations):out=dispatcher(q,k,v)
 end.record();torch.cuda.synchronize()
 elapsed=time.perf_counter()-t
 # Packing отдельно от kernel+packing. Здесь uniform dense flatten не копирует
 # contiguous buffers; cu construction и output reshape остаются в общем времени.
 pstart,pend=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 pstart.record()
 for _ in range(a.iterations):
  packed=[x.reshape(-1,a.heads,a.head_dim).contiguous() for x in (q,k,v)]
  cu=torch.arange(2,device='cuda',dtype=torch.int32)*a.length
  unpacked=out.reshape_as(q)
 pend.record();torch.cuda.synchronize()
 report=dict(status='PASS',backend=dispatcher.report(),seed=44,shape=list(q.shape),iterations=a.iterations,warmup=3,
  cuda_kernel_plus_adapter_ms=start.elapsed_time(end)/a.iterations,wall_ms=elapsed*1000/a.iterations,
  packing_only_ms=pstart.elapsed_time(pend)/a.iterations,allocated=torch.cuda.memory_allocated(),
  reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),cpu_memory=process_memory(),
  note='kernel-only attribution requires CUDA profiler; CUDA events here include adapter launches')
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2));print(a.output)
