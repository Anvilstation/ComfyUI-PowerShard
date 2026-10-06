#!/usr/bin/env python3
"""Реальный native CLIP -> выбранные CUDA workers -> CPU conditioning/cache/idle.
Без скачивания весов/установки зависимостей. Reference только отдельным явным файлом.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser()
p.add_argument('--comfy',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--gpus',default='all');p.add_argument('--precision',choices=['fp16','int8_fp16'],default='int8_fp16')
p.add_argument('--weight-placement',choices=['gpu','cpu','ats'],default='gpu')
p.add_argument('--attention-backend',default='auto')
p.add_argument('--idle-policy',choices=['release','cpu_shards','keep'],default='release')
p.add_argument('--prefetch-blocks',type=int,choices=[0,1,2],default=0);p.add_argument('--numa-policy',choices=['none','auto','bind'],default='none')
p.add_argument('--prompt',default='Волны у каменного берега. Слышен шум моря.');p.add_argument('--image')
p.add_argument('--reference',help='safetensors с ключами cond и tags, полученный native encoder на тех же inputs')
p.add_argument('--rtol',type=float,default=.025);p.add_argument('--atol',type=float,default=.006)
p.add_argument('--output',default='reports/local-qwen');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
import torch
if not torch.cuda.is_available():
 (out/'acceptance.json').write_text(json.dumps({'status':'NOT_RUN','reason':'CUDA unavailable','torch':torch.__version__},indent=2))
 raise SystemExit('NOT_RUN: нет CUDA')
sys.path.insert(0,str(Path(a.comfy).resolve()));sys.argv=['powershard-qwen-accept','--disable-dynamic-vram','--use-pytorch-cross-attention']
import comfy.options
comfy.options.enable_args_parsing()
from powershard.config import DistributedConfig
from powershard.qwen import QwenConfig
from powershard.qwen_adapter import load_qwen
from powershard.conditioning_cache import content_hash
from safetensors.torch import save_file,load_file
cfg=DistributedConfig(tuple(a.gpus.split(',')),weight_placement=a.weight_placement,precision=a.precision,attention_backend=a.attention_backend,
    prefetch_blocks=a.prefetch_blocks,numa_policy=a.numa_policy,memory_policy='auto')
clip=load_qwen(a.checkpoint,cfg,QwenConfig(a.idle_policy),report_dir=out)
images=[]
if a.image:
 import numpy as np
 from PIL import Image
 images=[torch.from_numpy(np.array(Image.open(a.image).convert('RGB'),copy=True)).float().unsqueeze(0)/255.]
tokens=clip.tokenize(a.prompt,images=images)
try:
 results=[];times=[]
 for _ in range(2):
  start=time.perf_counter();result=clip.encode_from_tokens_scheduled(tokens)
  times.append(time.perf_counter()-start);results.append(result)
  assert result[0][0].device.type=='cpu' and torch.isfinite(result[0][0]).all()
 torch.testing.assert_close(results[0][0][0],results[1][0][0],rtol=0,atol=0)
 assert clip.cond_stage_model.last_encoding['cache_hit']
 cond,tags=results[0][0][0],results[0][0][1]['minimax_token_tags']
 reference={'status':'NOT_RUN','reason':'--reference not supplied'}
 if a.reference:
  ref=load_file(a.reference);torch.testing.assert_close(cond,ref['cond'],rtol=a.rtol,atol=a.atol)
  torch.testing.assert_close(tags,ref['tags'])
  reference=dict(status='PASS',rtol=a.rtol,atol=a.atol,max_abs=float((cond-ref['cond']).abs().max()))
 save_file({'cond':cond.contiguous(),'tags':tags.contiguous()},str(out/'conditioning.safetensors'))
 clip.clear_cache();third=clip.encode_from_tokens_scheduled(tokens)
 torch.testing.assert_close(cond,third[0][0],rtol=.003,atol=.003)
 report=dict(status='PASS',scope='real encoder; not H3 generation',input_sha256=content_hash(tokens),
    condition_sha256=content_hash(results[0]),config=clip.patcher.session.config.to_dict(),options=clip.patcher.session.role_options,
    shape=list(cond.shape),dtype=str(cond.dtype),cold_encoding_s=times[0],cache_hit_s=times[1],reference=reference,
    last_encoding=clip.cond_stage_model.last_encoding,history=clip.patcher.session.history)
 (out/'acceptance.json').write_text(json.dumps(report,indent=2));print('PASS:',out)
finally:clip.patcher.session.close()
