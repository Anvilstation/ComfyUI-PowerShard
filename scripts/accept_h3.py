#!/usr/bin/env python3
"""Прямой приёмочный forward реального checkpoint: три GPU, video+audio, warm/reload/cancel.
Качество текста/картинки этим тестом не проверяется: conditioning синтетическое.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('--comfy',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--profile');p.add_argument('--gpus');p.add_argument('--precision',choices=['fp16','int8_fp16']);p.add_argument('--backend',choices=['fsdp2','fsdp2_sequence']);p.add_argument('--output',default='reports/local-h3');p.add_argument('--lifecycle',action='store_true')
p.add_argument('--cpu-offload',action=argparse.BooleanOptionalAction,default=None);p.add_argument('--pin-memory',action=argparse.BooleanOptionalAction,default=None)
p.add_argument('--prefetch-blocks',type=int,choices=[0,1,2]);p.add_argument('--numa-policy',choices=['none','auto','bind'])
p.add_argument('--fp16-safe',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--debug-finite',action='store_true');p.add_argument('--text-magnitude',type=float,default=1.)
p.add_argument('--attention-backend',choices=['auto','sdpa','flash_attn','vllm_flash_attn','sageattention','math']);p.add_argument('--allow-fallback',action=argparse.BooleanOptionalAction,default=None)
p.add_argument('--memory-profile',choices=['custom','ram_min'],default=None)
p.add_argument('--workspace-mib',type=int,default=None)
p.add_argument('--stage-cache-mib',type=int,default=None)
a=p.parse_args()
import torch
from powershard.config import DistributedConfig
from powershard.runtime import Session
from powershard.checkpoint import Checkpoint
from powershard.patch_config import H3PatchConfig
ck=Checkpoint(a.checkpoint);cfg=ck.model_config()
options=json.loads(Path(a.profile).read_text()) if a.profile else {}
if a.gpus is not None:options['gpu_ids']=tuple(a.gpus.split(','))
if a.backend is not None:options['backend']=a.backend
if a.precision is not None:options['precision']=a.precision
for key in ('cpu_offload','pin_memory','prefetch_blocks','numa_policy','attention_backend','allow_fallback','memory_profile','workspace_mib','stage_cache_mib'):
 if getattr(a,key) is not None:options[key]=getattr(a,key)
options.update(allow_unverified=True,release_after_sampling=False)
c=DistributedConfig(**options)
patch=H3PatchConfig(enabled=a.fp16_safe,debug_finite=a.debug_finite)
s=Session(str(ck.path),c,a.comfy,a.output,patch=patch)
g=torch.Generator().manual_seed(44)
video=torch.randn(1,24,2,4,4,generator=g);audio=torch.randn(1,32,2,8,generator=g);text=torch.randn(1,7,cfg['text_dim'],generator=g)*a.text_magnitude
kwargs={'transformer_options':{'sample_sigmas':torch.tensor([1.,.7,.3,0.])},'minimax_payload':{'seed':44}}
try:
 outputs=[]
 for step in range(3):
  context=s.call('preprocess_text',(text,),{})
  out=s.call('forward',([video,audio],torch.tensor([700.]),context),kwargs)
  assert torch.isfinite(context).all() and all(torch.isfinite(x).all() for x in out)
  assert len(out)==2 and out[0].shape==video.shape and out[1].shape==audio.shape
  if outputs:
   for x,y in zip(out,outputs[0]):torch.testing.assert_close(x,y,atol=2e-3,rtol=3e-3)
  outputs.append(out)
 if a.lifecycle:
  s.close()
  context=s.call('preprocess_text',(text,),{})
  try:
   def cancel():raise InterruptedError('Приёмочный тест отмены')
   s.call('forward',([video,audio],torch.tensor([700.]),context),kwargs,cancel=cancel)
  except InterruptedError:pass
  assert not s.running
  context=s.call('preprocess_text',(text*.9,),{})
  s.call('forward',([video,audio],torch.tensor([600.]),context),kwargs)
 from safetensors.torch import save_file
 Path(a.output).mkdir(parents=True,exist_ok=True)
 save_file({'video':outputs[0][0].contiguous(),'audio':outputs[0][1].contiguous()},str(Path(a.output,'latents.safetensors')))
 report={'status':'PASS','checkpoint':ck.identity(),'config':c.to_dict(),'patch':patch.to_dict(),'text_magnitude':a.text_magnitude,'input':'synthetic conditioning; not end-to-end generation',
         'seed':44,'repeated_forward':3,'lifecycle':bool(a.lifecycle),'rank_history':s.history}
 Path(a.output).mkdir(parents=True,exist_ok=True);Path(a.output,'acceptance.json').write_text(json.dumps(report,indent=2));print('PASS:',a.output)
finally:s.close()
