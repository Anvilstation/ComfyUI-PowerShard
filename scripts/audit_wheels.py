#!/usr/bin/env python3
"""Читает PyPI metadata: CPython 3.11, ppc64le, glibc <=2.31. Ничего не устанавливает."""
import argparse,json,urllib.request
from pathlib import Path
from packaging.tags import cpython_tags,compatible_tags
from packaging.utils import parse_wheel_filename
p=argparse.ArgumentParser();p.add_argument('--output',default='reports/local-portability311.json');a=p.parse_args()
platforms=['manylinux2014_ppc64le','linux_ppc64le']+[f'manylinux_2_{n}_ppc64le' for n in range(17,32)]
tags=set(cpython_tags((3,11),abis=['cp311'],platforms=platforms))|set(compatible_tags((3,11),interpreter='cp311',platforms=platforms))
versions={'safetensors':'0.6.2','tokenizers':'0.22.2','transformers':'4.57.3','comfy-kitchen':'0.2.34',
          'comfy-aimdo':'0.5.3','torchaudio':'2.11.0','sentencepiece':'0.2.1','av':'18.1.0','ray':'2.48.0','xfuser':'0.4.4','kernels':'0.17.0'}
rows={}
for package,version in versions.items():
 url=f'https://pypi.org/pypi/{package}/{version}/json'
 try:
  with urllib.request.urlopen(url,timeout=30) as response:meta=json.load(response)
  wheels=[]
  for file in meta['urls']:
   if file['filename'].endswith('.whl'):
    if tags.intersection(parse_wheel_filename(file['filename'])[3]):wheels.append(file['filename'])
  rows[package]={'version':version,'source':url,'matching_wheels':wheels,
                 'sdists':[file['filename'] for file in meta['urls'] if file['packagetype']=='sdist'],
                 'requires_python':meta['info']['requires_python'],'requires_dist':meta['info']['requires_dist']}
 except Exception as e:rows[package]={'version':version,'source':url,'error':str(e)}
report={'target':'CPython3.11 / ppc64le / Ubuntu20.04 glibc<=2.31','source_build_test':'NOT_RUN_ON_AC922','packages':rows,
        'note_ru':'Wheel tag match не доказывает runtime/CUDA kernel compatibility; отсутствие wheel не запрещает source build.'}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2));print(a.output)
