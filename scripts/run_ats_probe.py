#!/usr/bin/env python3
"""Явная изолированная проверка Unified Memory; ничего не меняет в PyTorch/ComfyUI.

Готовый binary передаётся через --binary. --build явно разрешает компиляцию
только небольшого probe во временном каталоге. Все результаты сохраняются.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='all')
    p.add_argument('--mib',type=int,default=64)
    p.add_argument('--timeout',type=int,default=120)
    p.add_argument('--build',action='store_true')
    p.add_argument('--binary')
    p.add_argument('--cuda-arch',default='native',help='nvcc -arch; native доступен в CUDA 12.4')
    p.add_argument('--output',default='reports/local-ats-memory.json')
    a=p.parse_args()
    if a.build == bool(a.binary):p.error('Выберите ровно один режим: --build или --binary')
    if not 1<=a.mib<=4096 or a.timeout<=0:p.error('mib 1..4096, timeout > 0')
    report=dict(status='NOT_RUN',probe='direct_system_malloc_and_cudaMallocManaged',results=[],
        application_allocator='UNCHANGED; PowerShard uses CPUOffloadPolicy')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='powershard-ats-probe-') as directory:
        try:
            from powershard.devices import resolve_gpu_selection
            devices=resolve_gpu_selection(a.gpus)
            report['selected_devices']=devices
            if a.build:
                nvcc=shutil.which('nvcc')
                if not nvcc:raise FileNotFoundError('nvcc не найден: укажите Toolkit в PATH или готовый --binary')
                binary=str(Path(directory)/'ats_probe')
                command=[nvcc,'-O2','-std=c++14','-arch='+a.cuda_arch,str(ROOT/'scripts/ats_memory_probe.cu'),'-o',binary]
                built=subprocess.run(command,capture_output=True,text=True,timeout=a.timeout)
                report['build']=dict(command=command,returncode=built.returncode,stderr=built.stderr[-8000:])
                if built.returncode:raise RuntimeError('Не удалось собрать probe; см. build.stderr')
            else:
                binary=str(Path(a.binary).resolve())
                if not Path(binary).is_file():raise FileNotFoundError(binary)
            for device in devices:
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device['uuid']
                try:
                    ran=subprocess.run([binary,str(a.mib)],env=env,capture_output=True,text=True,timeout=a.timeout)
                    result=json.loads(ran.stdout) if ran.returncode==0 else dict(status='FAIL',returncode=ran.returncode,reason=ran.stderr[-8000:])
                except (OSError,ValueError,subprocess.TimeoutExpired) as error:
                    result=dict(status='FAIL',reason=str(error))
                report['results'].append(dict(uuid=device['uuid'],user_id=device['user_id'],result=result))
            report['status']='FAIL' if any(row['result'].get('status')=='FAIL' for row in report['results']) else 'COMPLETED'
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
            report['reason']=str(error)
        out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0 if report['status']=='COMPLETED' else 2


if __name__=='__main__':raise SystemExit(main())
