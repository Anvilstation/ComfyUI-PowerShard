#!/usr/bin/env python3
"""Прямой приёмочный forward реального checkpoint: выбранные GPU, video+audio, warm/reload/cancel.
Качество текста/картинки этим тестом не проверяется: conditioning синтетическое.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('--comfy',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--profile');p.add_argument('--gpus');p.add_argument('--precision',choices=['fp16','int8_fp16']);p.add_argument('--output',default='reports/local-h3');p.add_argument('--lifecycle',action='store_true');p.add_argument('--ram-roundtrip',action='store_true')
p.add_argument('--weight-placement',choices=['gpu','cpu','ats']);p.add_argument('--sequence-mode',choices=['token','ulysses']);p.add_argument('--pin-memory',action=argparse.BooleanOptionalAction,default=None)
p.add_argument('--prefetch-blocks',type=int,choices=[0,1,2]);p.add_argument('--numa-policy',choices=['none','auto','bind'])
p.add_argument('--prefetch-policy',choices=['auto','manual'])
p.add_argument('--mlp-mode',choices=['auto','manual','off'],default='off');p.add_argument('--mlp-tokens',type=int,default=4096)
p.add_argument('--fp16-safe',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--debug-finite',action='store_true');p.add_argument('--text-magnitude',type=float,default=1.)
p.add_argument('--frames',type=int,default=2);p.add_argument('--latent-height',type=int,default=8);p.add_argument('--latent-width',type=int,default=8)
p.add_argument('--audio-tokens',type=int,default=5);p.add_argument('--text-tokens',type=int,default=7)
p.add_argument('--sequence-comm-dtype',choices=['fp16','fp32'])
p.add_argument('--attention-backend',choices=['auto','sdpa','flash_attn','vllm_flash_attn','sageattention','math']);p.add_argument('--allow-fallback',action=argparse.BooleanOptionalAction,default=None)
a=p.parse_args()
if min(a.frames,a.latent_height,a.latent_width,a.audio_tokens,a.text_tokens)<1:p.error('input dimensions must be positive')
import torch
from powershard.config import DistributedConfig
from powershard.runtime import Session
from powershard.checkpoint import Checkpoint
from powershard.patch_config import H3PatchConfig
ck=Checkpoint(a.checkpoint);cfg=ck.model_config()
options=json.loads(Path(a.profile).read_text()) if a.profile else {}
if a.gpus is not None:options['gpu_ids']=tuple(a.gpus.split(','))
options['backend']='fsdp2_sequence'
options['precision']='int8_fp16' if ck.quantization() else 'fp16'
if a.precision is not None:options['precision']=a.precision
if a.prefetch_policy is not None:options['memory_policy']=a.prefetch_policy
for key in ('weight_placement','sequence_mode','sequence_comm_dtype','pin_memory','prefetch_blocks','numa_policy','attention_backend','allow_fallback'):
 if getattr(a,key) is not None:options[key]=getattr(a,key)
options.update(release_after_sampling=False)
c=DistributedConfig(**options)
patch=H3PatchConfig(enabled=a.fp16_safe,debug_finite=a.debug_finite,mlp_chunk_mode=a.mlp_mode,mlp_chunk_tokens=a.mlp_tokens)
s=Session(str(ck.path),c,a.comfy,a.output,patch=patch)
g=torch.Generator().manual_seed(44)
video=torch.randn(1,cfg['latents_dim'],a.frames,a.latent_height,a.latent_width,generator=g)
audio=torch.randn(1,cfg['audio_latents_dim'],2,a.audio_tokens,generator=g)
text=torch.randn(1,a.text_tokens,cfg['text_dim'],generator=g)*a.text_magnitude
kwargs={'transformer_options':{'sample_sigmas':torch.tensor([1.,.7,.3,0.])},'minimax_payload':{'seed':44}}
from powershard.conditioning_cache import content_hash
input_sha256=content_hash(dict(video=video,audio=audio,text=text,timestep=torch.tensor([700.]),kwargs=kwargs))
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
 if a.ram_roundtrip:
  pids=[p.pid for p in s.processes]
  s.finish_sampling()
  assert s.running and s.idle_on_cpu and [p.pid for p in s.processes]==pids
  context=s.call('preprocess_text',(text,),{})
  out=s.call('forward',([video,audio],torch.tensor([700.]),context),kwargs)
  for x,y in zip(out,outputs[0]):torch.testing.assert_close(x,y,atol=2e-3,rtol=3e-3)
  if not c.cpu_offload:
   resumed=[m['metrics']['phase_cache'] for row in s.history for m in row['ranks'] if m.get('metrics',{}).get('phase_cache',{}).get('resumed_from_ram')]
   assert len(resumed)==len(pids) and all(m['checkpoint_weight_reads_on_resume']==0 for m in resumed)
 if a.lifecycle:
  s.close()
  context=s.call('preprocess_text',(text,),{})
  try:
   def cancel():raise InterruptedError('Приёмочный тест отмены')
   s.call('forward',([video,audio],torch.tensor([700.]),context),kwargs,cancel=cancel)
  except InterruptedError:pass
  assert s.running # already-cancelled prompt must not unload retained weights
  context=s.call('preprocess_text',(text*.9,),{})
  s.call('forward',([video,audio],torch.tensor([600.]),context),kwargs)
 from safetensors.torch import save_file
 Path(a.output).mkdir(parents=True,exist_ok=True)
 save_file({'video':outputs[0][0].contiguous(),'audio':outputs[0][1].contiguous()},str(Path(a.output,'latents.safetensors')))
 report={'status':'PASS','checkpoint':ck.identity(),'config':c.to_dict(),'patch':patch.to_dict(),'text_magnitude':a.text_magnitude,'input':'synthetic conditioning; not end-to-end generation',
         'input_sha256':input_sha256,'seed':44,'repeated_forward':3,'lifecycle':bool(a.lifecycle),'ram_roundtrip':bool(a.ram_roundtrip),'rank_history':s.history}
 report['input_shapes']={'video':list(video.shape),'audio':list(audio.shape),'text':list(text.shape)}
 Path(a.output).mkdir(parents=True,exist_ok=True);Path(a.output,'acceptance.json').write_text(json.dumps(report,indent=2));print('PASS:',a.output)
finally:s.close()
