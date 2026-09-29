#!/usr/bin/env python3
"""Проверка native sampler/model interface на CPU с локальным H3, НЕ distributed runtime."""
import argparse,copy,json,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--comfy',required=True);p.add_argument('--output',default='reports/comfy-contract-cpu.json');a=p.parse_args()
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));sys.path.insert(0,str(Path(a.comfy).resolve()))
sys.argv=['test','--cpu','--disable-dynamic-vram','--use-pytorch-cross-attention']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.sample, comfy.samplers, comfy.nested_tensor, comfy.supported_models
from comfy.ldm.minimax.model import MiniMaxH3Model
from powershard.operations import Operations
from powershard.attention import install_attention
from powershard.config import DistributedConfig
from powershard.comfy_adapter import RemoteH3,DiffusionProxy,PowerShardPatcher
from powershard.wire import write_payload,read_payload
conf=dict(hidden_size=32,num_layers=2,token_refiner_num_layers=1,num_attention_heads=7,attention_head_dim=8,ffn_hidden_size=64,text_dim=24,time_embed_dim=4,adaln_curve_grid=17,rope_inv_freq_len=1,latents_dim=24,audio_latents_dim=32)
torch.set_num_threads(1);torch.manual_seed(40)
config=DistributedConfig(reserve_gib=0,allow_unverified=True)
with torch.device('meta'):net=MiniMaxH3Model(**conf,dtype=torch.float16,device='meta',operations=Operations)
net.to_empty(device='cpu').eval().requires_grad_(False)
for k,v in net.named_parameters():
 v.copy_(torch.ones_like(v) if 'norm' in k else torch.randn(v.shape).to(v.dtype)*.01)
net.adaln_t_table.fill_(.1);net.rope.inv_freq.fill_(.01);install_attention(net,config)
from powershard.runtime import Session
class LocalContractSession(Session):
 # Тестовый transport для проверки Comfy API, не входит в пользовательский backend.
 def __init__(self):
  super().__init__(None,config,a.comfy,Path(a.output).resolve().parent)
  self.calls=0;self.active=False
 @property
 def running(self):return self.active
 def close(self):self.active=False
 def control(self,command):
  assert command=='end_run'
 def save_run_summary(self,context):
  super().save_run_summary(context)
  path=self.report_dir/('run-'+context['run_id']+'.json')
  data=json.loads(path.read_text());data['execution_mode']='CPU_LOCAL_CONTRACT_NO_CUDA_NO_FSDP';data['performance_evidence']=False
  path.write_text(json.dumps(data,indent=2))
 def call(self,command,args,kwargs,cancel=None):
  self.calls+=1;self.active=True
  with tempfile.TemporaryDirectory() as d:
   write_payload(d,{'args':args,'kwargs':kwargs});v=read_payload(d)
  if command=='preprocess_text':return net.preprocess_text_embeds(*v['args'],**v['kwargs'])
  return net(*v['args'],**v['kwargs'])
session=LocalContractSession()
mc=comfy.supported_models.MiniMaxH3(dict(conf,image_model='minimax_h3',disable_unet_model_creation=True,dtype=torch.float16));mc.manual_cast_dtype=torch.float16
model=RemoteH3(mc,device=torch.device('cpu'));model.diffusion_model=DiffusionProxy(session,conf)
patcher=PowerShardPatcher(model,torch.device('cpu'),torch.device('cpu'),size=1)
clone=patcher.clone();clone.model_options['transformer_options']['minimax_h3_sigma_shift_video']=9.
assert 'minimax_h3_sigma_shift_video' not in patcher.model_options['transformer_options']
assert not list(model.diffusion_model.parameters())
condition=[[torch.randn(1,5,24),{}]]
video=torch.zeros(1,24,2,4,4);audio=torch.zeros(1,32,2,5)
latent=comfy.nested_tensor.NestedTensor([video,audio])
results=[]
for seed in (5,6,5):
 noise=comfy.sample.prepare_noise(latent,seed)
 with torch.no_grad():
  out=comfy.sample.sample(patcher,noise,2,1.,'euler','simple',condition,condition,latent,seed=seed,disable_pbar=True)
 streams=out.unbind();assert len(streams)==2 and streams[0].shape==video.shape and streams[1].shape==audio.shape
 assert all(torch.isfinite(x).all() for x in streams)
 results.append([x.clone() for x in streams])
for x,y in zip(results[0],results[2]):torch.testing.assert_close(x,y,rtol=0,atol=0)
assert not torch.equal(results[0][0],results[1][0])
# Запрет conditioning до первого RPC.
before=session.calls
try:
 noise=comfy.sample.prepare_noise(latent,5)
 with torch.no_grad():comfy.sample.sample(patcher,noise,2,1.,'euler','simple',condition,[[condition[0][0],{'gligen':('unsupported',None)}]],latent,seed=5,disable_pbar=True)
except ValueError:pass
else:raise AssertionError('gligen не отклонён')
assert session.calls==before
report={'status':'PASS','device':'CPU x86_64','sampler':'native ComfyUI Euler/simple CFG=1','steps':2,'seeds':[5,6,5],
        'audio_video':'PASS','clone_model_options':'PASS','seed_change_repeat':'PASS','reject_patch_before_RPC':'PASS',
        'transport':'TEST_LOCAL_CONTRACT_ONLY','distributed_CUDA':'NOT_RUN','real_checkpoint':'NOT_RUN'}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
