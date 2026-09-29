#!/usr/bin/env python3
"""Явная диагностика installed providers и изолированные CUDA probes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from powershard.devices import visible_inventory,resolve_gpu_selection
from powershard.web_api import provider_inventory
from powershard.attention_policy import prepare_policy
from powershard.config import DistributedConfig
p=argparse.ArgumentParser();p.add_argument('--gpus',default='all')
p.add_argument('--backend',choices=['auto','sdpa','flash_attn','vllm_flash_attn','sageattention','math'],default='auto')
p.add_argument('--allow-fallback',action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--checkpoint');p.add_argument('--output',default='reports/attention-probe.json');a=p.parse_args()
report={'providers':provider_inventory()}
try:
 inventory=visible_inventory()
 if not inventory:report.update(status='NOT_RUN',reason='Нет доступных CUDA GPU',POWER9_V100_custom_wheel='NOT_RUN_ON_AC922')
 else:
  config=DistributedConfig(gpu_ids=a.gpus,attention_backend=a.backend,allow_fallback=a.allow_fallback)
  devices=resolve_gpu_selection(config.gpu_ids,inventory)
  report.update(status='PASS',selected_devices=devices,world_size=len(devices),policy=prepare_policy(config,devices,a.checkpoint))
except (RuntimeError,ValueError,OSError) as error:report.update(status='FAIL',reason=str(error))
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False));sys.exit(1 if report['status']=='FAIL' else 0)
