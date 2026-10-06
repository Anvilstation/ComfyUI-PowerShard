#!/usr/bin/env python3
"""Отправка одного подготовленного API workflow локальной ComfyUI. По умолчанию только план."""
import argparse,json,time,urllib.request,urllib.parse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from powershard.reporting import redact
p=argparse.ArgumentParser();p.add_argument('workflow');p.add_argument('--server',default='http://127.0.0.1:8188');p.add_argument('--submit',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--steps',type=int);p.add_argument('--output',default='reports/local-workflow.json');a=p.parse_args()
from powershard.workflow_migration import migrate_api
graph,migrations=migrate_api(json.loads(Path(a.workflow).read_text()))
for n in graph.values():
 if n['class_type']=='RandomNoise' and a.seed is not None:n['inputs']['noise_seed']=a.seed
 if n['class_type']=='BasicScheduler' and a.steps is not None:n['inputs']['steps']=a.steps
print(json.dumps({'server':a.server,'workflow':a.workflow,'submit':a.submit,'seed':a.seed,'steps':a.steps},indent=2))
if not a.submit:raise SystemExit('План готов; для выполнения добавьте --submit')
url=a.server.rstrip('/')
request=urllib.request.Request(url+'/prompt',data=json.dumps({'prompt':graph}).encode(),headers={'Content-Type':'application/json'})
start=time.perf_counter()
with urllib.request.urlopen(request,timeout=60) as r:reply=json.load(r)
if reply.get('node_errors'):raise RuntimeError(reply['node_errors'])
id=reply['prompt_id'];print('prompt_id:',id,flush=True)
while True:
 with urllib.request.urlopen(url+'/history/'+urllib.parse.quote(id),timeout=60) as r:history=json.load(r)
 if id in history:
  item=history[id]
  result={'workflow':redact(graph),'prompt_id':id,'wall_latency_s':time.perf_counter()-start,'history':redact(item),
          'stage_timings_ru':'См. output/powershard worker reports. wall latency включает очередь; не использовать для сравнения без idle сервера.'}
  Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print('Результат:',a.output)
  if item.get('status',{}).get('status_str')!='success':raise SystemExit(1)
  break
 time.sleep(1)
