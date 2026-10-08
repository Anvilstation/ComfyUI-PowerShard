#!/usr/bin/env python3
"""Парные cases: меняется одна ось. По умолчанию сохраняет графы/план; не отправляет задания."""
import argparse,copy,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from powershard.reporting import redact
p=argparse.ArgumentParser();p.add_argument('workflow');p.add_argument('--axis',choices=['sequence','mlp','placement','attention','spectrum','wire','prefetch'],required=True)
p.add_argument('--output-dir',default='reports/local-matrix');p.add_argument('--repeats',type=int,default=2)
p.add_argument('--submit',action='store_true');p.add_argument('--server',default='http://127.0.0.1:8188');a=p.parse_args()
if a.repeats<1:p.error('repeats >= 1')
from powershard.workflow_migration import migrate_api
base,_=migrate_api(json.loads(Path(a.workflow).read_text()));out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
values=dict(sequence=['token','ulysses'],mlp=[1024,4096,8192,16384,'off','auto'],
    placement=['gpu','cpu','ats'],attention=['sdpa','vllm_flash_attn','flash_attn'],spectrum=[False,True],wire=['fp32','fp16'],prefetch=[0,1,2])[a.axis]
loaders=[n for n in base.values() if n['class_type']=='PowerShardH3Loader']
if len(loaders)!=1:p.error('Нужен один H3 loader для однозначного сравнения')
config_id=str(loaders[0]['inputs']['config'][0])
visited=set()
while base[config_id]['class_type']=='PowerShardConfigTuning':
 if config_id in visited:p.error('Cycle in ConfigTuning chain')
 visited.add(config_id);config_id=str(base[config_id]['inputs']['config'][0])
if base[config_id]['class_type']!='PowerShardConfig':p.error('Expected PowerShardConfig below the optional ConfigTuning chain')
manifest={'axis':a.axis,'base_graph':redact(base),'cases':[],
    'note_ru':'Для принудительного full execution запускайте ComfyUI с --cache-none. Это может пересоздать loaders: cold/warm определять по worker load/history, не по номеру повтора. Warm RPC отдельно accept_h3/accept_qwen. Графы inputs сохраняются явно этим инструментом.'}
for value in values:
 graph=copy.deepcopy(base);cfg=graph[config_id]['inputs']
 if a.axis=='sequence':cfg['sequence_mode']=value
 elif a.axis=='placement':cfg['weight_placement']=value
 elif a.axis=='attention':cfg['attention_backend']=value
 elif a.axis in ('wire','prefetch'):
  tuning_id=str(graph[next(k for k,n in graph.items() if n['class_type']=='PowerShardH3Loader')]['inputs']['config'][0])
  if graph[tuning_id]['class_type']!='PowerShardConfigTuning':
   key=str(max(int(k) for k in graph if k.isdigit())+1)
   graph[key]=dict(class_type='PowerShardConfigTuning',inputs=dict(config=[tuning_id,0],reserve_gib=2.,prefetch_blocks=0,
       numa_policy='auto',strict_attention=False,allow_host_wrappers=False,pin_memory=True,
       sequence_comm_dtype='fp32',prefetch_policy='auto'))
   next(n for n in graph.values() if n['class_type']=='PowerShardH3Loader')['inputs']['config']=[key,0]
   tuning_id=key
  if a.axis=='wire':graph[tuning_id]['inputs']['sequence_comm_dtype']=value
  else:graph[tuning_id]['inputs'].update(prefetch_blocks=value,prefetch_policy='manual')
 elif a.axis=='mlp':
  if not any(n['class_type']=='PowerShardH3MLP' for n in graph.values()):p.error('Для MLP matrix добавьте PowerShardH3MLP в MODEL chain')
  for n in graph.values():
   if n['class_type']=='PowerShardH3MLP':
    n['inputs']['mode']=value if isinstance(value,str) else 'manual'
    if isinstance(value,int):n['inputs']['chunk_tokens']=value
 else:
  spectrum=[(k,n) for k,n in graph.items() if n['class_type']=='PowerShardSpectrum']
  if not spectrum:
   consumers=[n for n in graph.values() if n['class_type'] in ('BasicGuider','BasicScheduler','CFGGuider')]
   links={tuple(n['inputs']['model']) for n in consumers}
   if len(links)!=1:p.error('Guider/scheduler должны использовать один MODEL')
   key=str(max(map(int,graph))+1);graph[key]={'class_type':'PowerShardSpectrum','inputs':{'model':list(links.pop()),'enabled':value,'history_device':'cpu','history_mib':512}}
   for n in consumers:n['inputs']['model']=[key,0]
  else:
   for _,n in spectrum:n['inputs']['enabled']=value
 name=f'{a.axis}-{str(value).lower()}';path=out/(name+'.api.json');path.write_text(json.dumps(graph,indent=2,ensure_ascii=False))
 case={'value':value,'workflow':str(path),'results':[]};manifest['cases'].append(case)
 if a.submit:
  for repeat in range(a.repeats):
   result_path=out/f'{name}-repeat{repeat}.json'
   subprocess.run([sys.executable,str(ROOT/'scripts/run_api_workflow.py'),str(path),'--server',a.server,'--submit','--output',str(result_path)],check=True)
   record=json.loads(result_path.read_text());history=record['history']
   cached=[m[1].get('nodes',[]) for m in history.get('status',{}).get('messages',[]) if m[0]=='execution_cached']
   sampler_ids={k for k,n in graph.items() if n['class_type'] in ('SamplerCustomAdvanced','KSampler')}
   was_cached=bool(sampler_ids.intersection(x for group in cached for x in group))
   case['results'].append(dict(repeat=repeat,file=str(result_path),status='CACHED_NOT_BENCHMARK' if was_cached else 'EXECUTED_CHECK_WORKER_REPORTS'))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False));print(out/'manifest.json')
