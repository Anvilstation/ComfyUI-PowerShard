#!/usr/bin/env python3
"""Родной H3 ComfyUI, малые случайные веса CPU. Не тест checkpoint качества или CUDA."""
import argparse,copy,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--comfy',required=True);p.add_argument('--output',default='reports/native-cpu.json');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root));sys.path.insert(0,str(Path(a.comfy).resolve()))
sys.argv=['powershard-test','--cpu','--disable-dynamic-vram','--use-pytorch-cross-attention']
import comfy.options
comfy.options.enable_args_parsing()
import torch
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
from powershard.operations import Operations
from powershard.attention import install_attention
from powershard.config import DistributedConfig
from powershard.checkpoint import infer_h3_config
from powershard.source_guard import verify_comfy
verify_comfy(a.comfy)
torch.set_num_threads(1);torch.manual_seed(44)
conf=dict(hidden_size=32,num_layers=2,token_refiner_num_layers=1,num_attention_heads=7,attention_head_dim=8,
          ffn_hidden_size=64,text_dim=24,time_embed_dim=4,adaln_curve_grid=17,rope_inv_freq_len=1)
with torch.device('meta'):
 net=MiniMaxH3Model(**conf,dtype=torch.float16,device='meta',operations=Operations)
net.to_empty(device='cpu').eval().requires_grad_(False)
for name,param in net.named_parameters():
 if 'norm' in name:param.copy_(torch.ones_like(param))
 else:param.copy_(torch.randn(param.shape).to(param.dtype)*.02)
net.rope.inv_freq.fill_(.01);net.adaln_t_table.copy_(torch.randn_like(net.adaln_t_table)*.1)
custom=copy.deepcopy(net);install_attention(custom,DistributedConfig())
video=torch.randn(1,24,2,4,4).half();audio=torch.randn(1,32,2,5).half();text=torch.randn(1,7,24).half();t=torch.tensor([700.])
# Both streams, uneven tokens, per-row masks, image+audio keyframes, mixed text tags.
cases=[('text',{},{}),('masked',{}, {'denoise_mask':torch.rand(1,1,2,4,4),'audio_denoise_mask':torch.rand(1,1,2,5)}),
       ('fl2va',{'keyframes':[{'resolved_frame_index':0,'latent':video[:,:,:1].float(),'audio_latent':audio[:,:,:,:2].float()}],
                  'cond_video_latents':[video[:,:,:1].float()],'cond_audio_latents':[audio[:,:,:,:2].float()],
                  'text_token_tags':torch.tensor([[1,0,0,1,1,1,1]]),'seed':77},{}),
       ('ref2va',{'refs':[{'kind':'image','latent_h':4,'latent_w':4,'latent':video[:,:,:1].float()}],
                  'cond_video_latents':[video[:,:,:1].float()],'seed':79},{}),
       ('audio_scale',{'audio_scale':4.}, {})]
results=[]
with torch.no_grad():
 for name,payload,extra in cases:
  ref=net([video.clone(),audio.clone()],t,text.clone(),minimax_payload=payload,**extra)
  got=custom([video.clone(),audio.clone()],t,text.clone(),minimax_payload=payload,**extra)
  errors=[]
  for r,g in zip(ref,got):
   torch.testing.assert_close(g,r,atol=2e-3,rtol=3e-3);errors.append((g-r).abs().max().item())
  results.append({'case':name,'status':'PASS','max_abs_video_audio':errors})
 refined=custom.preprocess_text_embeds(text.clone())
 assert refined.shape==(1,7,32)
 ref=custom([video.clone(),audio.clone()],t,text.clone())
 got=custom([video.clone(),audio.clone()],t,refined)
 for r,g in zip(ref,got):torch.testing.assert_close(g,r,atol=2e-3,rtol=3e-3)
 for mode in ['no_grad','inference_mode']:
  with getattr(torch,mode)():
   out=custom([video.clone(),audio.clone()],t,text.clone())
   assert all(torch.isfinite(x).all() for x in out)
# Реальная архитектура из header: создание только meta, никаких 40 GB weights.
header=json.loads((root/'models/minimax_h3_fl2va_pruned_bf16.safetensors.header.json').read_text())
realconf=infer_h3_config(header)
with torch.device('meta'):
 meta=MiniMaxH3Model(**realconf,dtype=torch.float16,device='meta',operations=Operations)
expected=set(header)
actual=set(meta.state_dict())
assert expected==actual,(expected-actual,actual-expected)
for name,tensor in meta.state_dict().items():assert list(tensor.shape)==header[name]['shape'],name
report={'device':'CPU x86_64','torch':torch.__version__,'weights':'малые случайные, НЕ реальные checkpoint weights',
        'FSDP':'NOT_RUN','CUDA_NCCL':'NOT_RUN','native_H3_model':results,'preprocess_text':'PASS',
        'small_unsharded_modes':['eval+no_grad','eval+inference_mode'],
        'full_H3_meta_key_shape_match':'PASS','config':realconf}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
