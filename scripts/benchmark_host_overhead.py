#!/usr/bin/env python3
"""CPU overhead provider identity; не выдавать за CUDA speedup."""
import argparse,json,sys,time,statistics,subprocess,types,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from powershard.web_api import provider_stamp
p=argparse.ArgumentParser();p.add_argument('--output',default='reports/provider-stamp-current.json');p.add_argument('--repeats',type=int,default=10);p.add_argument('--baseline-commit');a=p.parse_args()
if a.repeats<2:p.error('repeats >= 2')
times=[]
for _ in range(a.repeats):
 t=time.perf_counter();provider_stamp();times.append((time.perf_counter()-t)*1000)
result=dict(scope='CPU provider_stamp overhead only; no CUDA kernel',cold_ms=times[0],warm_ms=times[1:],warm_median_ms=statistics.median(times[1:]))
if a.baseline_commit:
 if not re.fullmatch('[0-9a-fA-F]{7,40}',a.baseline_commit):p.error('Нужен конкретный commit SHA')
 source=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'show',a.baseline_commit+':powershard/web_api.py'],text=True)
 module=types.ModuleType('powershard.benchmark_baseline');module.__package__='powershard'
 exec(compile(source,'baseline/web_api.py','exec'),module.__dict__)
 before=[]
 for _ in range(a.repeats):
  t=time.perf_counter();module.provider_stamp();before.append((time.perf_counter()-t)*1000)
 result['baseline']=dict(commit=a.baseline_commit,cold_ms=before[0],warm_ms=before[1:],warm_median_ms=statistics.median(before[1:]))
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
