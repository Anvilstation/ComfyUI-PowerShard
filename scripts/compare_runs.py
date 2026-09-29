#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--rtol',type=float,default=.003);p.add_argument('--atol',type=float,default=.002);a=p.parse_args()
import torch
from safetensors.torch import load_file
for key in ['seed','input','checkpoint']:
 x=json.loads(Path(a.reference,'acceptance.json').read_text())[key];y=json.loads(Path(a.candidate,'acceptance.json').read_text())[key]
 if key=='checkpoint':
  # Сравниваются одна и та же модель+dtype, разные FSDP/SP. Между checkpoint-format нужны иные критерии качества.
  x=x['header_sha256'];y=y['header_sha256']
 if x!=y:raise SystemExit('Несовпадение условий сравнения: '+key)
r=load_file(str(Path(a.reference,'latents.safetensors')));t=load_file(str(Path(a.candidate,'latents.safetensors')))
for k in r:
 error=(r[k].float()-t[k].float()).abs();print(k,'max_abs=',error.max().item(),'rms=',error.square().mean().sqrt().item())
 torch.testing.assert_close(t[k],r[k],rtol=a.rtol,atol=a.atol)
print('PASS; допуски:',a.atol,a.rtol)
