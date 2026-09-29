#!/usr/bin/env python3
"""Минимальный воспроизводимый preflight toolchain. Систему не изменяет."""
import json,platform,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from powershard.diagnostics import command
out={'python_3_11':sys.version_info[:2]==(3,11),'architecture':platform.machine(),
     'gcc':command(['g++','-dumpfullversion']), 'nvcc':command(['nvcc','--version']),
     'torch_source_audit':{'minimum_cuda':'12.1','minimum_gcc':'11.3','cuda_12_4_configure_gate':'passes version gate only; full build NOT_RUN'},
     'note_ru':'На Ubuntu 20.04 штатные Python 3.8 / GCC 9 не отвечают исходным требованиям torch 2.12. Это не запрет уже установленной кастомной сборки.'}
print(json.dumps(out,indent=2,ensure_ascii=False))
