#!/usr/bin/env python3
"""Послойная BF16-source/FP32-reference vs FP16 проверка без целой H3 в RAM/GPU."""
import argparse,json,sys,copy
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--comfy',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--block',type=int,default=0);p.add_argument('--tokens',type=int,default=7);p.add_argument('--output',default='reports/local-block.json');a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(a.comfy).resolve()));sys.argv=['test','--cpu','--disable-dynamic-vram']
import comfy.options;comfy.options.enable_args_parsing()
import torch
from safetensors import safe_open
from comfy.ldm.minimax.model import DiTBlock,rope_rotation_table
from powershard.operations import Operations
from powershard.attention import install_attention
from powershard.config import DistributedConfig
from powershard.checkpoint import Checkpoint
c=Checkpoint(a.checkpoint);cfg=c.model_config()
if c.quantization():raise SystemExit('Этот тест сравнивает BF16 source; для ConvRot используйте отдельный checkpoint и probe_three')
with torch.device('meta'):
 ref=DiTBlock(cfg['hidden_size'],cfg['num_attention_heads'],cfg['attention_head_dim'],cfg['ffn_hidden_size'],cfg['time_embed_dim'],1e-5,1e-5,apply_silu=False,adaln_dtype=torch.float32,dtype=torch.float32,device='meta',operations=Operations)
 test=DiTBlock(cfg['hidden_size'],cfg['num_attention_heads'],cfg['attention_head_dim'],cfg['ffn_hidden_size'],cfg['time_embed_dim'],1e-5,1e-5,apply_silu=False,adaln_dtype=torch.float32,dtype=torch.float16,device='meta',operations=Operations)
ref.to_empty(device='cpu').eval().requires_grad_(False);test.to_empty(device='cpu').eval().requires_grad_(False)
with safe_open(str(c.path),framework='pt',device='cpu') as f,torch.no_grad():
 for name,param in ref.named_parameters():
  src=f.get_tensor(f'blocks.{a.block}.{name}')
  if not torch.isfinite(src).all():raise FloatingPointError(name)
  target=dict(test.named_parameters())[name]
  if target.dtype==torch.float16 and src.float().abs().max()>65504:raise FloatingPointError('FP16 overflow: '+name)
  param.copy_(src.float());target.copy_(src.to(target.dtype))
 table=f.get_tensor('adaln_t_table')[[128,512,768]]
install_attention(ref,DistributedConfig());install_attention(test,DistributedConfig())
from powershard.fp16_safe import FiniteTracker,install_safe_operations,wrap_safe_block
tracker=FiniteTracker(debug=True);install_safe_operations(test,tracker,True);wrap_safe_block(test,512)
g=torch.Generator().manual_seed(123);x=torch.randn(a.tokens,cfg['hidden_size'],generator=g)*.25
rope=rope_rotation_table(torch.zeros(a.tokens,cfg['rope_inv_freq_len']*6),torch.float32)
with torch.no_grad():
 r=ref(x.clone(),table,[(0,a.tokens,0)],rope)
 tracker.begin('cpu');t=test(x.float(),table,[(0,a.tokens,0)],rope.float());tracker.finish(t)
error=(r-t).abs();report={'checkpoint':c.identity(),'block':a.block,'tokens':a.tokens,'reference':'CPU FP32 from BF16','compute':'FP16 Safe scaled GEMMs; FP32 residual/SiLU/norm',
                         'max_abs':error.max().item(),'rms_abs':error.square().mean().sqrt().item(),'finite':bool(torch.isfinite(t).all()),
                         'note_ru':'Синтетические входы одного блока. Это не доказательство устойчивости всех diffusion steps.'}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
torch.testing.assert_close(t,r,rtol=.03,atol=.03)
