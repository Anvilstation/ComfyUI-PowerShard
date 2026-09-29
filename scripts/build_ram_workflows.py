#!/usr/bin/env python3
"""Только новые RAM-примеры; остальные workflows остаются неизменными."""
import copy
import json
from build_workflows import ROOT, SPECS, make, node, ui

SPECS['PowerShardConfig'][2].extend(['weight_placement','sequence_mode','sequence_comm_dtype','memory_profile','workspace_mib','stage_cache_mib'])


def main():
    for name,variant,gpus,precision,backend in (
        ('fl2va_ram_min_int8','fl2va','0,1,2','int8_fp16','fsdp2'),
        ('fl2va_ram_min_all_sequence','fl2va','all','int8_fp16','fsdp2_sequence'),
        ('ref2va_ram_min_fp16','ref2va','all','fp16','fsdp2')):
        graph=make(variant,precision,backend,True)
        graph['1']['inputs'].update(gpu_ids=gpus,timeout_s=1800,attention_backend='sdpa',allow_fallback=True,
            memory_policy='auto',weight_placement='cpu',sequence_mode='token',sequence_comm_dtype='fp32',
            memory_profile='ram_min',workspace_mib=256,stage_cache_mib=64)
        graph['21']=copy.deepcopy(graph['1'])
        graph['3']=node('PowerShardH3QwenLoader',checkpoint='qwen3vl_32b_minimax_h3_'+('bf16' if precision=='fp16' else 'int8_convrot')+'.safetensors',
            config=['21',0],precision=precision,idle_policy='cpu_shards',cache_mib=256,mlp_chunk_mode='auto',mlp_chunk_tokens=4096)
        graph['19']['inputs'].update(mlp_chunk_mode='auto',mlp_chunk_tokens=4096)
        graph['6']['inputs']['length']=25
        graph['12']['inputs'].update(preserve_qwen_cpu_shards=True,clear_conditioning_cache=False)
        graph['20']=node('SaveImage',images=['14',0],filename_prefix='PowerShard_PNG/'+name)
        graph['17']['inputs']['filename_prefix']='video/'+name
        for suffix,data in [('api',graph),('ui',ui(graph))]:
            (ROOT/'workflows'/f'{name}.{suffix}.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
