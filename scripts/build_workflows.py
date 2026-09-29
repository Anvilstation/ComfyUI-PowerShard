#!/usr/bin/env python3
"""Детерминированные API/UI graphs по проверенным native node interfaces."""
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPECS={
'PowerShardConfig':([],['POWERSHARD_CONFIG'],['gpu_ids','backend','precision','reserve_gib','timeout_s','allow_unverified','release_after_sampling','cpu_offload','pin_memory','prefetch_blocks','numa_policy','attention_backend','allow_fallback','memory_policy']),
'PowerShardH3Loader':([('config','POWERSHARD_CONFIG')],['MODEL'],['checkpoint']),
'PowerShardH3FP16Patcher':([('model','MODEL')],['MODEL'],['enabled','fp16_safe','debug_finite','mlp_chunk_tokens','mlp_chunk_mode']),
'PowerShardH3QwenLoader':([('config','POWERSHARD_CONFIG')],['CLIP'],['checkpoint','precision','idle_policy','cache_mib','mlp_chunk_mode','mlp_chunk_tokens']),
'PowerShardSpectrum':([('model','MODEL')],['MODEL'],['enabled','history_device','history_mib','degree','warmup','tail','max_forecast','history_size','ridge','blend','audio_blend']),
'PowerShardH3TextEncoder':([],['CLIP'],['checkpoint','placement']),
'VAELoader':([],['VAE'],['vae_name']),
'MiniMaxH3ImageToVideo':([('clip','CLIP'),('vae','VAE'),('first_frame','IMAGE')],['CONDITIONING','LATENT'],['prompt','width','height','length']),
'MiniMaxH3ReferenceToVideo':([('clip','CLIP'),('vae','VAE'),('audio_vae','VAE')],['CONDITIONING','LATENT'],['prompt','width','height','length','ref_image_size']),
'BasicGuider':([('model','MODEL'),('conditioning','CONDITIONING')],['GUIDER'],[]),
'RandomNoise':([],['NOISE'],['noise_seed']),
'KSamplerSelect':([],['SAMPLER'],['sampler_name']),
'BasicScheduler':([('model','MODEL')],['SIGMAS'],['scheduler','steps','denoise']),
'SamplerCustomAdvanced':([('noise','NOISE'),('guider','GUIDER'),('sampler','SAMPLER'),('sigmas','SIGMAS'),('latent_image','LATENT')],['LATENT','LATENT'],[]),
'PowerShardRelease':([('samples','LATENT')],['LATENT','STRING'],['preserve_qwen_cpu_shards','clear_conditioning_cache']),
'LTXVSeparateAVLatent':([('av_latent','LATENT')],['LATENT','LATENT'],[]),
'VAEDecodeTiled':([('samples','LATENT'),('vae','VAE')],['IMAGE'],['tile_size','overlap','temporal_size','temporal_overlap']),
'VAEDecodeAudio':([('samples','LATENT'),('vae','VAE')],['AUDIO'],[]),
'CreateVideo':([('images','IMAGE'),('audio','AUDIO')],['VIDEO'],['fps']),
'SaveVideo':([('video','VIDEO')],[],['filename_prefix','format','codec']),
'SaveImage':([('images','IMAGE')],[],['filename_prefix']),
'LoadImage':([],['IMAGE','MASK'],['image'])}

def node(kind,**inputs):return {'class_type':kind,'inputs':inputs}

def make(variant='fl2va',precision='fp16',backend='fsdp2',cpu_offload=False):
 checkpoint=f'minimax_h3_{variant}_pruned_'+('bf16' if precision=='fp16' else 'int8_convrot')+'.safetensors'
 graph={
 '1':node('PowerShardConfig',gpu_ids='0,1,2',backend=backend,precision=precision,reserve_gib=2.,timeout_s=600,allow_unverified=False,release_after_sampling=True,cpu_offload=cpu_offload,pin_memory=True,prefetch_blocks=0,numa_policy='none'),
 '2':node('PowerShardH3Loader',checkpoint=checkpoint,config=['1',0]),
 '19':node('PowerShardH3FP16Patcher',model=['2',0],enabled=True,fp16_safe=True,debug_finite=False,mlp_chunk_tokens=512),
 '3':node('PowerShardH3TextEncoder',checkpoint='qwen3vl_32b_minimax_h3_bf16.safetensors',placement='cpu_fp32'),
 '4':node('VAELoader',vae_name='minimax_h3_video_vae_fp16.safetensors'),
 '5':node('VAELoader',vae_name='minimax_h3_audio_vae_fp32.safetensors'),
 '6':node('MiniMaxH3ImageToVideo',clip=['3',0],vae=['4',0],prompt='Волны у каменного берега. Камера неподвижна. Слышен шум моря.',width=768,height=768,length=5),
 '7':node('BasicGuider',model=['19',0],conditioning=['6',0]),
 '8':node('RandomNoise',noise_seed=44),
 '9':node('KSamplerSelect',sampler_name='euler'),
 '10':node('BasicScheduler',model=['19',0],scheduler='simple',steps=20,denoise=1.),
 '11':node('SamplerCustomAdvanced',noise=['8',0],guider=['7',0],sampler=['9',0],sigmas=['10',0],latent_image=['6',1]),
 '12':node('PowerShardRelease',samples=['11',0]),
 '13':node('LTXVSeparateAVLatent',av_latent=['12',0]),
 '14':node('VAEDecodeTiled',samples=['13',0],vae=['4',0],tile_size=512,overlap=64,temporal_size=64,temporal_overlap=8),
 '15':node('VAEDecodeAudio',samples=['13',1],vae=['5',0]),
 '16':node('CreateVideo',images=['14',0],audio=['15',0],fps=24.),
 '17':node('SaveVideo',video=['16',0],filename_prefix='video/PowerShard_H3',format='auto',codec='auto')}
 if variant=='ref2va':
  graph['6']=node('MiniMaxH3ReferenceToVideo',clip=['3',0],vae=['4',0],audio_vae=['5',0],prompt='Волны у берега. Шум моря.',width=768,height=768,length=5,ref_image_size='match')
 return graph

def ui(graph):
 nodes=[];byid={};links=[]
 for index,(key,n) in enumerate(graph.items()):
  ins,outs,widgets=SPECS[n['class_type']]
  obj=dict(id=int(key),type=n['class_type'],pos=[(index//5)*400,(index%5)*260],size=[345,210],flags={},order=index,mode=0,
           inputs=[dict(name=name,type=dt,link=None) for name,dt in ins if name in n['inputs']],
           outputs=[dict(name=t,type=t,links=[],slot_index=i) for i,t in enumerate(outs)],
           properties={'Node name for S&R':n['class_type']},widgets_values=[n['inputs'][w] for w in widgets if w in n['inputs']])
  if n['class_type']=='RandomNoise':obj['widgets_values'].append('fixed')
  if n['class_type']=='LoadImage':obj['widgets_values'].append('image')
  nodes.append(obj);byid[key]=obj
 for key,n in graph.items():
  for idx,inp in enumerate(byid[key]['inputs']):
   val=n['inputs'][inp['name']];lid=len(links)+1;source,slot=val
   links.append([lid,int(source),slot,int(key),idx,inp['type']]);inp['link']=lid;byid[source]['outputs'][slot]['links'].append(lid)
 return dict(last_node_id=max(x['id'] for x in nodes),last_link_id=len(links),nodes=nodes,links=links,groups=[],config={},extra={'ds':{'scale':.75,'offset':[30,30]}},version=.4)

def main():
 for name,variant,precision,backend,offload in [('fl2va_fp16','fl2va','fp16','fsdp2',False),('fl2va_int8','fl2va','int8_fp16','fsdp2',False),('fl2va_sequence','fl2va','fp16','fsdp2_sequence',False),('ref2va_fp16','ref2va','fp16','fsdp2',False),('fl2va_fp16_offload','fl2va','fp16','fsdp2',True),('fl2va_int8_offload','fl2va','int8_fp16','fsdp2',True),('ref2va_int8','ref2va','int8_fp16','fsdp2',False)]:
  graph=make(variant,precision,backend,offload)
  for suffix,data in [('api',graph),('ui',ui(graph))]:
   (ROOT/'workflows'/f'{name}.{suffix}.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
 # FL2VA с настоящим первым кадром. Загрузите свой reference.png в ComfyUI input.
 graph=make();graph['18']=node('LoadImage',image='reference.png');graph['6']['inputs']['first_frame']=['18',0]
 for suffix,data in [('api',graph),('ui',ui(graph))]:(ROOT/'workflows'/f'fl2va_image.{suffix}.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))

 # Старые workflows выше остаются без новых полей. Новые примеры — обычный
 # native sampler с сохранением исходных PNG перед видеокодеком.
 for name,gpus,attention,offload in [('fl2va_all_sdpa','all','sdpa',False),
     ('fl2va_all_vllm_offload','all','vllm_flash_attn',True),('fl2va_subset_math','5,2,0','math',False)]:
  graph=make(precision='int8_fp16',cpu_offload=offload)
  graph['1']['inputs'].update(gpu_ids=gpus,attention_backend=attention,allow_fallback=True)
  graph['20']=node('SaveImage',images=['14',0],filename_prefix='PowerShard_PNG/'+name)
  for suffix,data in [('api',graph),('ui',ui(graph))]:
   (ROOT/'workflows'/f'{name}.{suffix}.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))

 # 0.4 examples; старые graphs и позиции widgets выше не изменяются.
 for name,precision,backend,offload,spectrum,frames in [
  ('fl2va_qwen_int8','int8_fp16','fsdp2',False,False,25),
  ('fl2va_qwen_fp16_offload','fp16','fsdp2',True,False,25),
  ('fl2va_qwen_int8_offload','int8_fp16','fsdp2',True,False,25),
  ('fl2va_qwen_sequence','int8_fp16','fsdp2_sequence',True,False,25),
  ('fl2va_long_auto','int8_fp16','fsdp2',True,False,361),
  ('fl2va_spectrum_euler','int8_fp16','fsdp2',True,True,25),
  ('fl2va_spectrum_few_step','int8_fp16','fsdp2',True,True,25)]:
  graph=make(precision=precision,backend=backend,cpu_offload=offload)
  graph['1']['inputs'].update(attention_backend='sdpa',allow_fallback=True,memory_policy='auto',timeout_s=1800)
  graph['19']['inputs'].update(mlp_chunk_mode='auto',mlp_chunk_tokens=4096)
  graph['21']=copy.deepcopy(graph['1']) # тот же Config class, собственные настройки encoder
  graph['3']=node('PowerShardH3QwenLoader',checkpoint='qwen3vl_32b_minimax_h3_'+('bf16' if precision=='fp16' else 'int8_convrot')+'.safetensors',
     config=['21',0],precision=precision,idle_policy='cpu_shards' if offload else 'release',cache_mib=256,mlp_chunk_mode='auto',mlp_chunk_tokens=4096)
  graph['6']['inputs']['length']=frames
  graph['12']['inputs'].update(preserve_qwen_cpu_shards=True,clear_conditioning_cache=False)
  graph['20']=node('SaveImage',images=['14',0],filename_prefix='PowerShard_PNG/'+name)
  if spectrum:
   graph['22']=node('PowerShardSpectrum',model=['19',0],enabled=True,history_device='cpu',history_mib=512,
      degree=2,warmup=3,tail=1,max_forecast=1,history_size=6,ridge=.001,blend=.5,audio_blend=0.)
   graph['7']['inputs']['model']=['22',0];graph['10']['inputs']['model']=['22',0]
   if name.endswith('few_step'):graph['10']['inputs']['steps']=4
  graph['17']['inputs']['filename_prefix']='video/'+name
  for suffix,data in [('api',graph),('ui',ui(graph))]:
   (ROOT/'workflows'/f'{name}.{suffix}.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))


if __name__ == "__main__":main()
