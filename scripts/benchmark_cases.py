#!/usr/bin/env python3
"""Парные cases: меняется одна ось. По умолчанию сохраняет графы/план; не отправляет задания."""
import argparse,copy,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from powershard.reporting import redact
p=argparse.ArgumentParser();p.add_argument('workflow');p.add_argument('--axis',choices=['backend','mlp','offload','attention','spectrum','memory'],required=True)
p.add_argument('--output-dir',default='reports/local-matrix');p.add_argument('--repeats',type=int,default=2)
p.add_argument('--submit',action='store_true');p.add_argument('--server',default='http://127.0.0.1:8188');a=p.parse_args()
if a.repeats<1:p.error('repeats >= 1')
base=json.loads(Path(a.workflow).read_text());out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
values=dict(backend=['fsdp2','fsdp2_sequence'],mlp=[1024,4096,8192,16384,'off','auto'],
    offload=[False,True],attention=['sdpa','vllm_flash_attn','flash_attn'],spectrum=[False,True],memory=['custom','ram_min'])[a.axis]
loaders=[n for n in base.values() if n['class_type']=='PowerShardH3Loader']
if len(loaders)!=1:p.error('Нужен один H3 loader для однозначного сравнения')
config_id=loaders[0]['inputs']['config'][0]
base_config=base[config_id]['inputs']
if a.axis in ('offload','mlp') and base_config.get('memory_profile')=='ram_min':
 p.error('ram_min переопределяет offload/MLP: используйте --axis memory или исходный workflow с memory_profile=custom')
if a.axis=='offload' and base_config.get('weight_placement') in ('cpu','ats'):
 p.error('weight_placement=cpu/ats принудительно включает offload: для сравнения on/off установите gpu в исходном workflow')
manifest={'axis':a.axis,'base_graph':redact(base),'cases':[],
    'note_ru':'Для принудительного full execution запускайте ComfyUI с --cache-none. Это может пересоздать loaders: cold/warm определять по worker load/history, не по номеру повтора. Warm RPC отдельно accept_h3/accept_qwen. Графы inputs сохраняются явно этим инструментом.'}
for value in values:
 graph=copy.deepcopy(base);cfg=graph[config_id]['inputs']
 if a.axis=='backend':cfg['backend']=value
 elif a.axis=='offload':cfg['cpu_offload']=value
 elif a.axis=='attention':cfg['attention_backend']=value
 elif a.axis=='memory':cfg['memory_profile']=value
 elif a.axis=='mlp':
  for n in graph.values():
   if n['class_type']=='PowerShardH3FP16Patcher':
    n['inputs']['mlp_chunk_mode']=value if isinstance(value,str) else 'manual'
    if isinstance(value,int):n['inputs']['mlp_chunk_tokens']=value
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
